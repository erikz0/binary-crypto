#!/usr/bin/env python3
"""
Walk-forward optimiser – **time-weighted P&L per trade** edition
• One objective: mean( pnl_dollars / risk_dollars / hold_bars )
  across *ALL* trades from train + CV + OOS.
• Keeps one rolling “best” checkpoint per (symbol, TF) and the
  top-3 champions overall.
• No Numba, no GA sampler – just plain TPE.
"""

from __future__ import annotations
import json, logging, math, os, random, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import pickle
from torch.utils.data import Subset
import traceback

import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml, optuna
from optuna.samplers import TPESampler

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, Sort
from alpaca.data.enums import CryptoFeed

import itertools

# ────────────────────────────── dirs / logging ─────────────────────────────
ROOT            = Path.cwd()
TRAINED_MODELS  = ROOT / "trained_models";  TRAINED_MODELS.mkdir(exist_ok=True)
PARAMS_DIR      = ROOT / "best_params";     PARAMS_DIR.mkdir(exist_ok=True)
CHAMP_DIR       = ROOT / "champions";       CHAMP_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────── config load ───────────────────────────────
CFG = yaml.safe_load(Path("config.yaml").read_text())
sector_list = CFG.get("sectors", [])
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")

# constants
EPS = 1e-8

_global_counter = itertools.count()   # import itertools

RAW_CACHE: dict[tuple[str, str], pd.DataFrame] = {}   #  (sym, tf) → raw bars
FEAT_CACHE: dict[str, dict[int, pd.DataFrame]] = defaultdict(dict)

# ─────────────────────── fetch crypto bars ───────────────────────────
def _call_with_retry(fn, req, label, max_retries, base_sleep):
    for attempt in range(1, max_retries + 1):
        try:
            #log.info("[%s] attempt %d/%d", label, attempt, max_retries)
            return fn(req, feed=CryptoFeed.US)
        except HTTPError as err:
            code = getattr(err.response, "status_code", None)
            if code == 429 and attempt < max_retries:
                wait = base_sleep * (2 ** (attempt - 1)) + random.random() * base_sleep
                log.warning("[%s] rate-limited (429); retrying in %.1fs", label, wait)
                time.sleep(wait)
                continue
            log.error("[%s] HTTP %s – giving up", label, code)
            raise
    raise RuntimeError(f"[{label}] exceeded {max_retries} retries")

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)

def fetch(sym: str, tf: TimeFrame, start, end, max_retries=5, base_sleep=1.5) -> pd.DataFrame:
    """
    Fetch historical crypto bars for `sym` (@ timeframe `tf`), UTC.
    Uses on-disk caching so repeat calls with identical params load locally.
    """

    # normalize symbol to BASE/QUOTE form
    api_sym = sym

    # ensure tz-aware UTC timestamps
    start_ts = pd.to_datetime(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    end_ts = pd.to_datetime(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    # format for filename: YYYYMMDDTHHMMSS±ZZZZ
    start_str = start_ts.strftime("%Y%m%dT%H%M%S%z")
    end_str   = end_ts.strftime("%Y%m%dT%H%M%S%z")

    cache_file = CACHE_DIR / f"{api_sym.replace('/', '')}_{tf.value}_{start_str}_{end_str}.pkl"
    # if cache_file.exists():
    #     log.info(f"Loading from cache: {cache_file.name}")
    #     return pd.read_pickle(cache_file)

    log.info(f"Cache miss — fetching {api_sym} {start_ts} → {end_ts}")
    cli = CryptoHistoricalDataClient(api_key=CFG["api_key"], secret_key=CFG["secret_key"])
    req = CryptoBarsRequest(
        symbol_or_symbols=[api_sym],
        timeframe=tf,
        start=start_ts,
        end=end_ts,
        sort=Sort.ASC
    )
    bar_resp = _call_with_retry(cli.get_crypto_bars, req, "CRYPTO_BARS", max_retries, base_sleep)[api_sym]

    df = pd.DataFrame(
        {
            "open"       : [b.open        for b in bar_resp],
            "high"       : [b.high        for b in bar_resp],
            "low"        : [b.low         for b in bar_resp],
            "close"      : [b.close       for b in bar_resp],
            "volume"     : [b.volume      for b in bar_resp],
            "vwap"       : [b.vwap        for b in bar_resp],
            "trade_count": [b.trade_count for b in bar_resp],
        },
        index=[b.timestamp for b in bar_resp],
    )
    df.index = pd.to_datetime(df.index).tz_convert("UTC")

    # save for next time
    df.to_pickle(cache_file)
    #log.info(f"Saved cache: {cache_file.name}")

    df = df.dropna(subset=["open", "close"]).sort_index()

    return df

# ───────────────────────── feature builder ───────────────────────────────

def build_feats(df: pd.DataFrame, ma: int, sym: str) -> pd.DataFrame:
    if ma <= 0:
        raise ValueError("ma must be positive")
    d = df.copy()

    # moving average momentum features
    d['ma']    = d['close'].rolling(ma).mean()
    d['mom']   = d['ma'].diff()
    d['force'] = d['mom'].diff()
    d['jerk']  = d['force'].diff()

    # RSI
    delta = d['close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    d['rsi'] = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    # ATR
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - d['close'].shift()).abs(),
        (d['low']  - d['close'].shift()).abs()
    ], axis=1).max(axis=1)
    d['atr'] = tr.rolling(14).mean()

    # returns & EMA
    d['ret1']    = d['close'].pct_change()
    d['ret_ema'] = d['ret1'].ewm(span=10, adjust=False).mean()

    # MACD
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    d['macd']     = ema12 - ema26
    d['macd_sig'] = d['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bb_mid = d['close'].rolling(20).mean()
    bb_std = d['close'].rolling(20).std()
    d['boll_b'] = (d['close'] - (bb_mid - 2*bb_std)) / (4*bb_std)
    d['boll_w'] = 4*bb_std / bb_mid

    # Stochastic Oscillator
    low14  = d['low'].rolling(14).min()
    high14 = d['high'].rolling(14).max()
    d['stoch_k'] = 100*(d['close']-low14)/(high14-low14+EPS)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # volume & OBV
    d['vol_rel'] = d['volume']/d['volume'].rolling(20).mean()
    d['obv']     = (np.sign(d['close'].diff())*d['volume']).fillna(0).cumsum()

    # time-of-day encoding (24h cycle)
    t = d.index.tz_convert(None)
    mins = t.hour*60 + t.minute
    d['tod_sin'] = np.sin(2*np.pi*mins/1440)
    d['tod_cos'] = np.cos(2*np.pi*mins/1440)

    # VWAP deviation
    d['vwap_dev'] = (d['close'] - d['vwap']) / (d['vwap'] + EPS)

    return d.fillna(0)

FEATS_LIST = [
    "ma", "vwap", "trade_count", "vwap_dev",
    "mom", "force", "jerk", "rsi", "atr",
    "ret1", "ret_ema", "macd", "macd_sig",
    "boll_b", "boll_w", "stoch_k", "stoch_d",
    "vol_rel", "obv", "tod_sin", "tod_cos"
]

# ────────────────────────── constants ─────────────────────────────────────
FEE_PCT   = 0.00   # taker fee, each side
DTYPE_F32 = np.float32
SIM_INIT_CAP = 1_000_000
    
@dataclass
class _SimplePos:
    entry_idx: int
    entry_px: float
    tp_px: float
    sl_px: float
    qty: float
    risk_amt: float
    reserved_cost: float = 0  # capital reserved at entry

class SimpleBacktester:
    """
    Long-only backtester with dynamic TP/SL like RealisticBacktester,
    but simplified P&L handling:
      • static balance until exit (no capital reservation),
      • PnL applied at exit,
      • size by fixed risk fraction of initial capital,
      • dynamic TP/SL using rolling bands,
      • safeguards: min range, risk floor.
    """
    def __init__(
        self,
        *,
        horizon: int,
        min_range_pct: float = 0.10,
        initial_balance: float = SIM_INIT_CAP,
        risk_frac: float = 0.01,
        risk_floor_pct: float = 0.002,
        W: int = 500,
    ):
        self.horizon = horizon
        self.min_range_pct = min_range_pct
        self.initial_balance = initial_balance
        self.risk_frac = risk_frac
        self.risk_floor_pct = risk_floor_pct
        self.W = W

    def run(
        self,
        *,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        signals: np.ndarray,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute back‑test with dynamic TP/SL fractions and **multiple concurrent positions**.
        (Return signature and debug prints kept identical.)
        """
        n_bars = len(close_prices)
        cum_pnl = 0.0
        equity_curve = np.full(n_bars, self.initial_balance, dtype=DTYPE_F32)

        # per‑trade collections
        rets, holds, risks, r_mults, fees_paid = [], [], [], [], []

        positions: list[_SimplePos] = []          # ← now a list instead of a single pos
        H, W = self.horizon, self.W

        # pre‑compute dynamic bands
        tp_frac_arr, sl_frac_arr = rolling_tp_sl(high_prices, low_prices,
                                                 close_prices, H, W)
        if debug:
            print(f"[BT] dynamic TP/SL with H={H}, W={W}, "
                  f"min_rng={self.min_range_pct:.2%}, "
                  f"risk_frac={self.risk_frac:.2%}, "
                  f"risk_floor={self.risk_floor_pct:.2%}")
            log.info(f"[BT] → {signals.sum()}/{len(signals)} signals")

        # stats counters
        total_potential_trades = 0
        total_skipped_trades_rng = 0
        total_skipped_trades_sl_tp_off = 0
        total_skipped_trades_no_risk = 0

        # ──────────────────────────────────────────────────────────────
        # MAIN BAR LOOP
        # ──────────────────────────────────────────────────────────────
        for t in range(n_bars):
            px_c, px_h, px_l = close_prices[t], high_prices[t], low_prices[t]
            equity_curve[t] = self.initial_balance + cum_pnl

            # 1️⃣ EXIT CHECK – evaluate every open position
            to_close: list[int] = []
            for idx, pos in enumerate(positions):
                age = t - pos.entry_idx
                hit_tp = px_h >= pos.tp_px
                hit_sl = px_l <= pos.sl_px
                timed  = age >= H

                if hit_tp or hit_sl or timed:
                    exit_px = (pos.tp_px if hit_tp else
                               pos.sl_px if hit_sl else px_c)
                    pnl = (exit_px - pos.entry_px) * pos.qty
                    r   = pnl / pos.risk_amt if pos.risk_amt else 0.0
                    fee = 0.0

                    cum_pnl += pnl
                    rets.append(pnl)
                    holds.append(age)
                    risks.append(pos.risk_amt)
                    r_mults.append(r)
                    fees_paid.append(fee)

                    if debug and t < 10:
                        kind = 'TP' if hit_tp else 'SL' if hit_sl else 'TIME'
                        print(f"[{kind}] idx={pos.entry_idx} hold={age} "
                              f"qty={pos.qty:.4f} pnl={pnl:+.2f} "
                              f"cum_pnl={cum_pnl:.2f}")

                    to_close.append(idx)

            # actually remove the closed positions (reverse order keeps indices valid)
            for idx in reversed(to_close):
                positions.pop(idx)

            # 2️⃣ ENTRY CHECK – we can always add a new position
            if signals[t] and t >= H and n_bars - t >= H:
                total_potential_trades += 1

                hi_t = high_prices[t - H + 1:t + 1].max()
                lo_t = low_prices[t - H + 1:t + 1].min()
                rng  = hi_t - lo_t

                # safeguard: minimum range
                if rng / px_c < self.min_range_pct:
                    total_skipped_trades_rng += 1
                    continue

                tp_frac = tp_frac_arr[t]
                sl_frac = sl_frac_arr[t]
                tp_px   = px_c + tp_frac * rng
                sl_px   = px_c - sl_frac * rng

                # safeguard: TP / SL must bracket price
                if sl_px >= px_c or tp_px <= px_c:
                    total_skipped_trades_sl_tp_off += 1
                    continue

                raw_risk = max(px_c - sl_px, px_c * self.risk_floor_pct)
                if raw_risk <= 0:
                    total_skipped_trades_no_risk += 1
                    continue

                risk_amt = self.initial_balance * self.risk_frac
                qty      = risk_amt / raw_risk

                new_pos = _SimplePos(
                    entry_idx=t,
                    entry_px=px_c,
                    tp_px=tp_px,
                    sl_px=sl_px,
                    qty=qty,
                    risk_amt=risk_amt,
                )
                positions.append(new_pos)

                if debug and t < 10:
                    print(f"[ENTRY] t={t} px={px_c:.2f} qty={qty:.3f} "
                          f"tp={tp_px:.2f} sl={sl_px:.2f} risk={risk_amt:.2f}")

        # 3️⃣ FORCE‑CLOSE anything still open at the final bar
        for pos in positions:
            exit_px = close_prices[-1]
            pnl = (exit_px - pos.entry_px) * pos.qty
            r   = pnl / pos.risk_amt if pos.risk_amt else 0.0
            fee = 0.0

            cum_pnl += pnl
            rets.append(pnl)
            holds.append((n_bars - 1) - pos.entry_idx)
            risks.append(pos.risk_amt)
            r_mults.append(r)
            fees_paid.append(fee)
        # positions list is discarded afterwards

        # ──────────────────────────────────────────────────────────────
        # PERFORMANCE STATISTICS (unchanged)
        # ──────────────────────────────────────────────────────────────
        trades      = len(r_mults)
        win_rate    = 100 * sum(r > 0 for r in r_mults) / trades if trades else 0.0
        avg_R_winners = np.mean([r for r in r_mults if r > 0]) if any(r > 0 for r in r_mults) else 0.0
        avg_R_losers  = np.mean([r for r in r_mults if r < 0]) if any(r < 0 for r in r_mults) else 0.0
        avg_fee_per_trade = np.mean(fees_paid) if fees_paid else 0.0
        expectancy_R = np.mean(r_mults) if trades else 0.0

        ending_balance = equity_curve[-1]
        pnl            = ending_balance - SIM_INIT_CAP
        percent_gains  = pnl / SIM_INIT_CAP
        sharpe         = (expectancy_R /
                          np.nanstd(r_mults, ddof=1)
                          if np.nanstd(r_mults, ddof=1) > 0 else np.nan)
        avg_hold_bars  = np.mean(holds)
        trade_rate     = trades / n_bars

        if debug:
            print(f"Trades {trades}")
            print(f"Trade Rate {trade_rate}")
            print(f"win_rate: {win_rate:.2f}%")
            print(f"avg_R_winners: {avg_R_winners:.2f}")
            print(f"avg_R_losers: {avg_R_losers:.2f}")
            print(f"avg_fee_per_trade: {avg_fee_per_trade:.4f}")
            print(f"expectancy_R: {expectancy_R:.4f}")
            print(f"Ending balance….. {ending_balance:,.2f}")
            print(f"PNL. {pnl:,.2f}")
            print(f'Percent gains: {percent_gains:.2%}')
            print(f'Sharpe: {sharpe:.2f}')
            print(f'Average Hold Bars: {avg_hold_bars}')
            print(f'total_potential_trades: {total_potential_trades}')
            print(f'total_skipped_trades_rng: {total_skipped_trades_rng}')
            print(f'total_skipped_trades_sl_tp_off: {total_skipped_trades_sl_tp_off}')
            print(f'total_skipped_trades_no_risk: {total_skipped_trades_no_risk}')

        # keep the original return signature
        return (
            trades,
            trade_rate,
            win_rate,
            avg_R_winners,
            avg_R_losers,
            expectancy_R,
            ending_balance,
            pnl,
            percent_gains,
            sharpe,
            avg_hold_bars,
        )
    
class RealisticBacktester:
    """
    Long-only backtester with dynamic TP/SL (rolling bands),
    but **realistic** capital management:
      • dynamic equity → risk_amt = risk_frac × equity
      • capital reservation per open tranche
      • pyramid entries (multiple concurrent positions)
      • skip entry if available cash < cost
    """
    def __init__(
        self,
        *,
        horizon: int,
        min_range_pct: float = 0.10,
        initial_balance: float = SIM_INIT_CAP,
        risk_frac: float = 0.001,
        risk_floor_pct: float = 0.002,
        W: int = 500,
    ):
        self.horizon = horizon
        self.min_range_pct = min_range_pct
        self.initial_balance = initial_balance
        self.risk_frac = risk_frac
        self.risk_floor_pct = risk_floor_pct
        self.W = W

    def run(
        self,
        *,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        signals: np.ndarray,
        debug: bool = False,
    ):
        """
        Execute backtest with dynamic TP/SL + realistic capital:
        Returns the same 11-tuple and prints as before.
        """
        n_bars = len(close_prices)
        # cash = free capital, reserved = sum of entry costs
        cash = self.initial_balance
        reserved = 0.0

        # track equity including unrealized: cash + market value of open pos
        equity_curve = np.full(n_bars, self.initial_balance, dtype=DTYPE_F32)

        rets, holds, risks, r_mults, fees_paid = [], [], [], [], []
        positions: list[_SimplePos] = []

        H, W = self.horizon, self.W
        tp_frac_arr, sl_frac_arr = rolling_tp_sl(
            high_prices, low_prices, close_prices, H, W
        )

        if debug:
            print(f"[BT] dynamic TP/SL with H={H}, W={W}, "
                  f"min_rng={self.min_range_pct:.2%}, "
                  f"risk_frac={self.risk_frac:.2%}, "
                  f"risk_floor={self.risk_floor_pct:.2%}")
            log.info(f"[BT] → {signals.sum()}/{len(signals)} signals")

        total_potential_trades = 0
        total_skipped_trades_rng = 0
        total_skipped_trades_sl_tp_off = 0
        total_skipped_trades_no_risk = 0

        for t in range(n_bars):
            px_c, px_h, px_l = close_prices[t], high_prices[t], low_prices[t]
            # compute unrealized PnL on open positions
            mv = sum((px_c - pos.entry_px) * pos.qty for pos in positions)
            equity = cash + reserved + mv
            equity_curve[t] = equity

            # 1) EXIT
            to_close = []
            for i, pos in enumerate(positions):
                age = t - pos.entry_idx
                hit_tp = px_h >= pos.tp_px
                hit_sl = px_l <= pos.sl_px
                timed  = age >= H
                if hit_tp or hit_sl or timed:
                    exit_px = pos.tp_px if hit_tp else pos.sl_px if hit_sl else px_c
                    pnl = (exit_px - pos.entry_px) * pos.qty
                    r   = pnl / pos.risk_amt if pos.risk_amt else 0.0
                    fee = 0.0

                    # release reserved + realize PnL
                    cash += pos.reserved_cost + pnl
                    reserved -= pos.reserved_cost

                    rets.append(pnl)
                    holds.append(age)
                    risks.append(pos.risk_amt)
                    r_mults.append(r)
                    fees_paid.append(fee)

                    if debug and t < 10:
                        kind = 'TP' if hit_tp else 'SL' if hit_sl else 'TIME'
                        print(f"[{kind}] idx={pos.entry_idx} hold={age} "
                              f"qty={pos.qty:.4f} pnl={pnl:+.2f} "
                              f"cum_equity={equity:.2f}")

                    to_close.append(i)

            for i in reversed(to_close):
                positions.pop(i)

            # 2) ENTRY
            if signals[t] and t >= H and n_bars - t >= H:
                total_potential_trades += 1

                hi_t = high_prices[t-H+1:t+1].max()
                lo_t = low_prices[t-H+1:t+1].min()
                rng  = hi_t - lo_t

                if rng / px_c < self.min_range_pct:
                    total_skipped_trades_rng += 1
                    continue

                tp_frac = tp_frac_arr[t]
                sl_frac = sl_frac_arr[t]
                tp_px   = px_c + tp_frac * rng
                sl_px   = px_c - sl_frac * rng

                if sl_px >= px_c or tp_px <= px_c:
                    total_skipped_trades_sl_tp_off += 1
                    continue

                raw_risk = max(px_c - sl_px, px_c * self.risk_floor_pct)
                if raw_risk <= 0:
                    total_skipped_trades_no_risk += 1
                    continue

                # dynamic risk on current equity
                risk_amt = equity * self.risk_frac
                qty      = risk_amt / raw_risk
                cost     = qty * px_c

                # skip if not enough free cash
                if cost > cash:
                    continue

                # reserve and open
                cash    -= cost
                reserved += cost
                positions.append(_SimplePos(
                    entry_idx=t,
                    entry_px=px_c,
                    tp_px=tp_px,
                    sl_px=sl_px,
                    qty=qty,
                    risk_amt=risk_amt,
                    reserved_cost=cost,
                ))

                if debug and t < 10:
                    print(f"[ENTRY] t={t} px={px_c:.2f} qty={qty:.3f} "
                          f"tp={tp_px:.2f} sl={sl_px:.2f} risk={risk_amt:.2f}")

        # 3) FORCE-CLOSE remaining
        for pos in positions:
            exit_px = close_prices[-1]
            pnl = (exit_px - pos.entry_px) * pos.qty
            r   = pnl / pos.risk_amt if pos.risk_amt else 0.0
            fee = 0.0

            cash    += pos.reserved_cost + pnl
            reserved -= pos.reserved_cost

            rets.append(pnl)
            holds.append((n_bars-1) - pos.entry_idx)
            risks.append(pos.risk_amt)
            r_mults.append(r)
            fees_paid.append(fee)

        # SUMMARY METRICS (exactly as before)
        trades          = len(r_mults)
        win_rate        = 100 * sum(r>0 for r in r_mults) / trades if trades else 0.0
        avg_R_winners   = np.mean([r for r in r_mults if r>0]) if any(r>0 for r in r_mults) else 0.0
        avg_R_losers    = np.mean([r for r in r_mults if r<0]) if any(r<0 for r in r_mults) else 0.0
        avg_fee_per_trade = np.mean(fees_paid) if fees_paid else 0.0
        expectancy_R    = np.mean(r_mults) if trades else 0.0

        ending_balance = cash  # after all forced closes, reserved=0
        pnl            = ending_balance - self.initial_balance
        percent_gains  = pnl / self.initial_balance
        sharpe         = expectancy_R / np.nanstd(r_mults, ddof=1) \
                          if np.nanstd(r_mults, ddof=1) > 0 else np.nan
        avg_hold_bars  = np.mean(holds)
        trade_rate     = trades / n_bars

        if debug:
            print(f"Trades {trades}")
            print(f"Trade Rate {trade_rate}")
            print(f"win_rate: {win_rate:.2f}%")
            print(f"avg_R_winners: {avg_R_winners:.2f}")
            print(f"avg_R_losers: {avg_R_losers:.2f}")
            print(f"avg_fee_per_trade: {avg_fee_per_trade:.4f}")
            print(f"expectancy_R: {expectancy_R:.4f}")
            print(f"Ending balance….. {ending_balance:,.2f}")
            print(f"PNL. {pnl:,.2f}")
            print(f'Percent gains: {percent_gains:.2%}')
            print(f'Sharpe: {sharpe:.2f}')
            print(f'Average Hold Bars: {avg_hold_bars}')
            print(f'total_potential_trades: {total_potential_trades}')
            print(f'total_skipped_trades_rng: {total_skipped_trades_rng}')
            print(f'total_skipped_trades_sl_tp_off: {total_skipped_trades_sl_tp_off}')
            print(f'total_skipped_trades_no_risk: {total_skipped_trades_no_risk}')

        return (
            trades,
            trade_rate,
            win_rate,
            avg_R_winners,
            avg_R_losers,
            expectancy_R,
            ending_balance,
            pnl,
            percent_gains,
            sharpe,
            avg_hold_bars,
        )    


# ────────────────────────── thin wrapper ────────────────────────────────────
def run_backtest(
    price_df : pd.DataFrame,
    signals  : np.ndarray,
    hp       : dict,
    *,
    horizon_override: Optional[int] = None,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Back‑test the TP‑first strategy.

    Returns
    -------
    (returns_per_trade, hold_durations, risk_allocations, equity_curve)
    """
    horizon      = hp["horizon"] if horizon_override is None else horizon_override
    min_range_pct= hp.get("min_range_pct", 0.10)
    tp_sl_lookback_window = hp.get("tp_sl_lookback_window", 500)

    bt = SimpleBacktester(
        horizon      = horizon,
        min_range_pct= min_range_pct,
        W            = tp_sl_lookback_window
    )

    return bt.run(
        close_prices = price_df["close"].to_numpy(dtype=DTYPE_F32),
        high_prices  = price_df["high"].to_numpy(dtype=DTYPE_F32),
        low_prices   = price_df["low"].to_numpy(dtype=DTYPE_F32),
        signals      = signals.astype(bool),
        debug        = debug,
    )

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
ROLL_SPAN_DEFAULT = 30         # 30 bars ≈ 1 trading day on 30‑min data


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _rolling_zscore(df: pd.DataFrame,
                    feats: list[str],
                    span: int = ROLL_SPAN_DEFAULT
                    ) -> tuple[np.ndarray, int]:
    """
    Row‑wise z‑score normalisation with an automatic warm‑up cut‑off.
    Returns (z‑scored array, first_valid_row_idx).
    """
    feat_df = df[feats]

    # rolling μ & σ
    mu = feat_df.rolling(span, min_periods=span).mean()
    sd = feat_df.rolling(span, min_periods=span).std(ddof=0).replace(0, 1.0)

    z = ((feat_df - mu) / sd).to_numpy(np.float32)

    valid = np.flatnonzero(~np.isnan(z).any(axis=1))
    if valid.size == 0:
        raise ValueError("No rows with valid rolling μ/σ – increase span or "
                         "provide longer data.")
    start = valid[0]
    return z[start:], start


def _windowise(z: np.ndarray, L: int) -> np.ndarray:
    """
    Convert a 2‑D (rows, feats) array into sliding windows
    shaped (windows, L, feats) with stride = 1 row.
    """
    win = np.lib.stride_tricks.sliding_window_view(z,
                                                   window_shape=L,
                                                   axis=0)
    return np.ascontiguousarray(win.transpose(0, 2, 1))  # (M, L, F)


# ──────────────────────────────────────────────────────────────────────
# Dynamic windows for an arbitrary slice
# ──────────────────────────────────────────────────────────────────────
def get_windows_dyn(df_slice: pd.DataFrame,
                    feats: list[str],
                    L: int,
                    roll_span: int = ROLL_SPAN_DEFAULT
                    ) -> np.ndarray:
    """
    Z‑score each feature column with a rolling μ/σ and
    return (windows, L, feats) ready for the network.
    """
    z, start_normal = _rolling_zscore(df_slice, feats, roll_span)
    windows = _windowise(z, L)

    return windows, start_normal


def rolling_tp_sl(highs: np.ndarray,
                   lows: np.ndarray,
                   closes: np.ndarray,
                   H: int,
                   W: int):
    """
    Compute dynamic TP/SL fractions using rolling medians of future scaled moves.

    Parameters
    ----------
    highs : np.ndarray, shape (T,)
    lows  : np.ndarray, shape (T,)
    closes: np.ndarray, shape (T,)
    H     : int
        Lookahead horizon for TP/SL hit detection and historical range.
    W     : int
        Window length for rolling calibration (in samples).

    Returns
    -------
    tp_frac_arr : np.ndarray, shape (T,)
        Rolling median of upward scaled moves.
    sl_frac_arr : np.ndarray, shape (T,)
        Rolling median of downward scaled moves.
    """
    # Series for rolling operations
    s_highs = pd.Series(highs)
    s_lows = pd.Series(lows)
    s_closes = pd.Series(closes)

    # 1) future max/min over next H bars
    fut_max = s_highs.shift(-1).rolling(H, min_periods=H).max()
    fut_min = s_lows.shift(-1).rolling(H, min_periods=H).min()

    # 2) historical high/low range over past H bars
    hist_hi = s_highs.rolling(H, min_periods=H).max()
    hist_lo = s_lows.rolling(H, min_periods=H).min()
    hist_range = (hist_hi - hist_lo)

    # avoid division by zero
    hist_range = hist_range.replace(0, np.nan)

    # 3) scaled future moves
    scaled_up = (fut_max - s_closes) / hist_range
    scaled_down = (s_closes - fut_min) / hist_range

    # 4) rolling medians for calibration
    tp_frac = scaled_up.rolling(W, min_periods=W).quantile(0.5)
    sl_frac = scaled_down.rolling(W, min_periods=W).quantile(0.5)

    # fill initial NaNs by forward-fill or with base 0
    tp_frac_arr = tp_frac.ffill().fillna(0).to_numpy()
    sl_frac_arr = sl_frac.ffill().fillna(0).to_numpy()

    return tp_frac_arr, sl_frac_arr


class TPSeqDS(Dataset):
    """
    Time-series dataset with dynamic TP/SL fractions that produce ~50/50 positives.
    """
    def __init__(
        self,
        X_cache: np.ndarray,   # (M, L, F)
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        L: int,
        H: int,
        W: int = 500,
        min_range_pct: float = 0.10,
        debug: bool = False
    ):
        # validations
        assert X_cache.ndim == 3, "X_cache must be (M, L, n_feats)"
        M = X_cache.shape[0]
        self.L, self.H = L, H
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.min_range_pct = min_range_pct

        # precompute rolling TP/SL fractions
        tp_frac_arr, sl_frac_arr = rolling_tp_sl(highs, lows, closes, H, W)

        #log.info(f'tp_frac_arr: {tp_frac_arr} sl_frac_arr: {sl_frac_arr} ')

        X_out, y_out = [], []
        cnt = { 'hist': 0, 'flat': 0, 'nohit': 0, 'keep': 0 }

        for i in range(M):
            entry_idx = i + L - 1
            # not enough history
            if entry_idx < H + W - 1:
                cnt['hist'] += 1
                continue

            close_t = closes[entry_idx]
            hi_t = highs[entry_idx - H + 1: entry_idx + 1].max()
            lo_t = lows[entry_idx - H + 1: entry_idx + 1].min()

            # volatility filter
            if (hi_t - lo_t) / close_t < min_range_pct:
                cnt['flat'] += 1
                continue

            # dynamic fractions
            tp_frac = tp_frac_arr[entry_idx]
            sl_frac = sl_frac_arr[entry_idx]

            tp_price = close_t + tp_frac * (hi_t - lo_t)
            sl_price = close_t - sl_frac * (hi_t - lo_t)

            fut_hi = highs[entry_idx + 1: entry_idx + 1 + H]
            fut_lo = lows[entry_idx + 1: entry_idx + 1 + H]

            hit_tp = np.any(fut_hi >= tp_price)
            hit_sl = np.any(fut_lo <= sl_price)
            if not (hit_tp or hit_sl):
                cnt['nohit'] += 1
                continue

            # assign label
            if hit_tp and hit_sl:
                t_tp = np.argmax(fut_hi >= tp_price)
                t_sl = np.argmax(fut_lo <= sl_price)
                y = 1.0 if t_tp < t_sl else 0.0
            else:
                y = 1.0 if hit_tp else 0.0

            X_out.append(X_cache[i])
            y_out.append(y)
            cnt['keep'] += 1

        if cnt['keep'] == 0:
            raise ValueError("No valid samples after TP/SL filtering!")

        self.X = torch.from_numpy(np.stack(X_out).astype(np.float32))
        self.y = torch.tensor(y_out, dtype=torch.float32)

        if debug:
            total = sum(cnt.values())
            pos_rate = float(self.y.mean())
            print(f"Windows={total}, Kept={cnt['keep']} ({cnt['keep']/total:.1%}), PosRate={pos_rate:.1%}")
            print(f"Skipped: hist={cnt['hist']}, flat={cnt['flat']}, nohit={cnt['nohit']}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegSimple(nn.Module):
    """
    Same backbone as before but returns *logits* for BCEWithLogitsLoss,
    with forced dropout on the final hidden state.
    """
    def __init__(self, c: int, h: int, L: int, d: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=c,
            hidden_size=h,
            num_layers=L,
            batch_first=True,
            dropout=d if L > 1 else 0.0,
        )
        self.dropout = nn.Dropout(d)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        h    = o[:, -1]
        h    = self.dropout(h)
        return self.fc(h).squeeze(-1)   # logits (B,)
    

def _make_splits(N: int, L: int, H: int, train_frac: float = 0.8) -> int:
    split = int(N * train_frac)
    train_end = split - H - L + 1
    if train_end <= 0:
        raise ValueError("Sequence/horizon too large for dataset length")
    return train_end


def train_one(
    df_train: pd.DataFrame,
    sym:      str,
    tf:       str,
    hp:       dict,
    feats:    List[str],
    device,
    debug: bool = True,
):
    # ───────────────────── offsets & params ────────────────────────
    L, H      = hp["seq_len"], hp["horizon"]
    roll_span = hp.get("roll_span", ROLL_SPAN_DEFAULT)
    patience  = hp.get("patience", 10)
    eps       = 1e-6

    tp_sl_lookback_window = hp["tp_sl_lookback_window"]
    min_rng   = hp.get("min_range_pct", 0.10)

    if debug:
        log.info(f"[{sym}/{tf}] ----------------  TRAIN START  ----------------")
        log.info(f"Dataset rows: {len(df_train):,}")
        log.info(f"seq_len / horizon {L} / {H}")
        log.info(f"tp_sl_lookback_window: {tp_sl_lookback_window}")
        log.info(f"min_range_pct:  {min_rng:.2%}")

    # ─────────────────── feature windows  ──────────────────────────
    X_all, start = get_windows_dyn(df_train, feats, L, roll_span=roll_span)
    n_win = len(X_all)
    train_end_win = _make_splits(n_win, L=L, H=H)

    if debug:
        log.info(f"Total windows…… {n_win:,}")
        log.info(f"Train / Val split {train_end_win:,} / {n_win - H - train_end_win:,}")
        log.info(f"Start idx for train: {start}")
        log.info(f"train_end_win: {train_end_win}")

    # ────────────────── price arrays aligned  ──────────────────────
    highs  = df_train["high" ].to_numpy(np.float32)[start:]
    lows   = df_train["low"  ].to_numpy(np.float32)[start:]
    closes = df_train["close"].to_numpy(np.float32)[start:]

    # ────────────────── dataset & loaders  ─────────────────────────
    ds_tr  = TPSeqDS(
        X_all[:train_end_win], highs, lows, closes,
        L, H, tp_sl_lookback_window, min_rng, debug=False
    )
    ds_val = TPSeqDS(
        X_all[train_end_win:n_win - H], highs[train_end_win:], lows[train_end_win:], closes[train_end_win:],
        L, H, tp_sl_lookback_window, min_rng, debug=False
    )

    # ────────────────── balance classes ───────────────────────────
    # Training set
    y_tr = ds_tr.y.numpy()
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_sel = min(len(pos_idx), len(neg_idx))
    sel_tr = np.concatenate([
        np.random.choice(pos_idx, n_sel, replace=False),
        np.random.choice(neg_idx, n_sel, replace=False),
    ])
    np.random.shuffle(sel_tr)
    ds_tr = Subset(ds_tr, sel_tr)

    # Validation set
    y_val = ds_val.y.numpy()
    pos_idx_val = np.where(y_val == 1)[0]
    neg_idx_val = np.where(y_val == 0)[0]
    n_sel_val = min(len(pos_idx_val), len(neg_idx_val))
    sel_val = np.concatenate([
        np.random.choice(pos_idx_val, n_sel_val, replace=False),
        np.random.choice(neg_idx_val, n_sel_val, replace=False),
    ])
    np.random.shuffle(sel_val)
    ds_val = Subset(ds_val, sel_val)

    if debug:
        tr_rate = y_tr[sel_tr].mean()
        vl_rate = y_val[sel_val].mean()
        log.info(f"Balanced samples…  train={len(ds_tr)} PosRate={tr_rate:.2%}  val={len(ds_val)} PosRate={vl_rate:.2%}")

    ld_tr  = DataLoader(ds_tr,
                        batch_size=hp.get("batch_size", 128),
                        shuffle=True, drop_last=True)
    ld_val = DataLoader(ds_val,
                        batch_size=hp.get("val_batch_size", 256),
                        shuffle=False)

    if debug and (len(ld_tr) == 0 or len(ld_val) == 0):
        log.error(f"[{sym}/{tf}] ERROR: Empty DataLoader! ld_tr={len(ld_tr)} ld_val={len(ld_val)}")
        raise RuntimeError("Empty DataLoader – check min_range_pct/horizon")

    # ───────────────── model / optimiser / scheduler ───────────────
    mdl = LSTMRegSimple(len(feats), hp["hidden_dim"],
                        hp["num_layers"], hp["dropout"]).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=hp["lr"] * 10,
        total_steps=len(ld_tr) * hp["epochs"],
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    bce = nn.BCEWithLogitsLoss()

    # ───────────────── training loop ───────────────────────────────
    best_wr    = 0.0
    stale_wr   = 0
    best_ckpt  = None
    best_epoch = 0

    for ep in range(1, hp["epochs"] + 1):
        t0 = time.time()
        mdl.train()
        train_loss = 0.0
        correct_tr = total_tr = 0

        for X, y in ld_tr:
            X, y = X.to(device), y.to(device)
            logits = mdl(X)
            loss = bce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            train_loss += loss.item()
            preds = (logits > 0).float()
            correct_tr += (preds == y).sum().item()
            total_tr   += y.numel()

        train_loss /= len(ld_tr)
        train_wr = correct_tr / total_tr

        # ───── validation ──────────────────────────────────────────
        mdl.eval()
        val_loss = 0.0
        correct_val = total_val = 0
        with torch.no_grad():
            for X, y in ld_val:
                X, y = X.to(device), y.to(device)
                logits = mdl(X)
                val_loss += bce(logits, y).item()
                preds = (logits > 0).float()
                correct_val += (preds == y).sum().item()
                total_val   += y.numel()
        val_loss /= len(ld_val)
        val_wr = correct_val / total_val

        # checkpoint based on winrate
        if val_wr > best_wr + eps:
            best_wr    = val_wr
            best_epoch = ep
            best_ckpt  = {
                "epoch": ep,
                "model": mdl.state_dict(),
                "optimizer": opt.state_dict(),
                "val_wr": val_wr,
                "hp": hp,
            }

        # patience on winrate
        stale_wr = 0 if val_wr > best_wr - eps else stale_wr + 1

        # ───── logging ─────────────────────────────────────────────
        if debug:
            lr = sched.get_last_lr()[0]
            star = "★" if ep == best_epoch else ""
            log.info(
                f"[{sym}/{tf}] E{ep:02d} "
                f"tr_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"tr_wr={train_wr:.3f} val_wr={val_wr:.3f} "
                f"lr={lr:.2e} t={time.time()-t0:.1f}s {star}"
            )

        if stale_wr >= patience:
            if debug:
                log.info(f"[{sym}/{tf}] Early stop @ epoch {ep} (winrate stale ≥ {patience})")
            break

    # ───────── restore best & calibrate ────────────────────────────
    mdl.load_state_dict(best_ckpt["model"])
    if debug:
        log.info(f"[{sym}/{tf}] Best epoch {best_ckpt['epoch']} val_wr={best_ckpt['val_wr']:.3f}")

    preds_val, y_val = [], []
    mdl.eval()
    with torch.no_grad():
        for X, y in ld_val:
            logits = mdl(X.to(device)).cpu().numpy()
            preds_val.append(logits)
            y_val.append(y.numpy())
    preds_val = np.concatenate(preds_val)
    y_val     = np.concatenate(y_val)

    if debug:
        pos_rate = (preds_val > 0).mean()
        log.info(f"[{sym}/{tf}] Calibration set size….. {len(preds_val):,}")
        log.info(f"[{sym}/{tf}] Logit > 0 rate………..... {pos_rate:.2%}")
        log.info(f"[{sym}/{tf}] ----------------  TRAIN END  ----------------")

    return mdl, preds_val


# ─────────────────── Optuna objective (single metric) ──────────────────────
BEST_PER_SERIES = defaultdict(lambda:-np.inf)
GLOBAL_TOP: List[Tuple[float,str,dict,dict]] = []  # (score, tag, sd, hp)


def slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug: bool = False) -> float:
    """Compute a composite score for a backtest slice, with detailed logging when debug=True."""

    final_score = .75 * expectancy_R + .25 * percent_gains

    return final_score


# ────────────────────────── infer & threshold ────────────────────────────
def infer_and_threshold(
    df: pd.DataFrame,
    threshold: float,
    models: list[torch.nn.Module],
    hp: dict,
    sym: str,
    tf: str,
    feats: List[str],
    device,
    debug: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run an ensemble of TP‑vs‑SL classifiers on `df`, calibrate the logits,
    take the mean probability, and apply a decision threshold θ.

    Returns
    -------
    sub   : pd.DataFrame            # slice of df aligned to predictions
    preds : np.ndarray, shape (M,)  # ensemble TP‑first probabilities
    sigs  : np.ndarray, shape (M,)  # boolean mask where preds > θ
    """
    # 1) build feature windows once (dynamic μ/σ)
    L         = hp["seq_len"]
    roll_span = hp.get("roll_span", ROLL_SPAN_DEFAULT)

    X_std, start = get_windows_dyn(df, feats, L, roll_span=roll_span)
    X_all = torch.from_numpy(X_std).float().to(device)

    # 2) calibrated predictions from each model
    preds_list = []
    for mdl in models:
        mdl.eval()
        with torch.no_grad():
            logits = mdl(X_all).cpu().numpy().ravel()

        probs = 1 / (1 + np.exp(-logits))

        preds_list.append(probs)

    # 3) ensemble – simple mean probability
    preds = np.mean(preds_list, axis=0)

    if debug:
        pct10, pct50, pct90 = np.percentile(preds, [10, 50, 90])
        log.info(f"[{sym}/{tf}] ens‑preds p10={pct10:.3f}  "
                 f"median={pct50:.3f}  p90={pct90:.3f}")

    # 4) align dataframe slice with predictions
    sub   = df.iloc[start+L-1 : start + len(preds)].reset_index(drop=True)

    # 5) signal mask
    sigs = preds > threshold

    if debug:
        log.info(f"[infer_and_threshold] θ={threshold:.3f} "
                 f"→ {sigs.sum()}/{len(sigs)} signals")

    return sub, preds, sigs


# ─────────────────────────────┐
#  Revised calibrate_threshold ↓
# ─────────────────────────────┘
def calibrate_threshold(
    df: pd.DataFrame,
    models: list[torch.nn.Module],
    hp: dict,
    sym: str,
    tf: str,
    feats: List[str],
    device,
    *,
    min_th: float,
    max_th: float,
    steps: int = 100,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Grid‑search θ ∈ [min_th, max_th] on the *ensemble* to maximise slice_score.
    Returns (best_threshold, best_score).
    """
    best_score, best_th = -np.inf, min_th
    thetas = np.linspace(min_th, max_th, steps)

    for θ in thetas:
        sub, preds, sigs = infer_and_threshold(
            df, θ, models, hp, sym, tf, feats, device, debug=debug
        )
        trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(sub, sigs, hp, debug=debug)
        score = slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug=debug)
        if score > best_score:
            best_score, best_th = score, θ
            if debug:
                log.info(f"→ θ={θ:.4f} gives slice_score={score:.6f}")

    return best_th, best_score

def stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug: bool = False) -> Dict[str, float]:
    """
    Compute performance statistics for a backtest.

    See docstring above for parameter descriptions.
    """

    return {
        "Trades"      : float(trades),
        "TradeRate"   : float(trade_rate),
        "WinRate"     : float(win_rate),
        "avg_R_winners": float(avg_R_winners),
        "avg_R_losers"      : float(avg_R_losers),
        "expectancy_R"   : float(expectancy_R),
        "ending_balance" : float(ending_balance),
        "pnl" : float(pnl),
        "percent_gains" : float(percent_gains * 100),
        "sharpe" : float(sharpe),
        "avg_hold_bars" : float(avg_hold_bars),
    }

# ───────────────────────────── objective ──────────────────────────────
def objective(
    trial,
    sym: str,
    tf: TimeFrame,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    raw_future: pd.DataFrame,
    device,
) -> float:
    """
    End‑to‑end Optuna objective.

    1. Sample hyper‑parameters
    2. Build / align features for TRAIN and TEST
    3. Train model on TRAIN slice
    4. Evaluate TRAIN, TEST, FUTURE (optional) slices
    5. Compute headline score with several penalty / bonus terms
    6. Persist model + score bookkeeping
    """

    # ─────────────────────────── 1) HYPER‑PARAM SAMPLING ────────────────────────────
    g = CFG["grid"]
    hp = {
        # model architecture
        "seq_len":      trial.suggest_categorical("seq_len", g["seq_len"]),
        "horizon":      trial.suggest_categorical("horizon", g["horizon"]),
        "num_layers":   trial.suggest_categorical("num_layers", g["num_layers"]),
        "hidden_dim":   trial.suggest_categorical("hidden_dim", g["hidden_dim"]),
        "dropout":      trial.suggest_categorical("dropout", g["dropout"]),

        # training / optimiser
        "epochs":       trial.suggest_categorical("epochs", g["epochs"]),
        "lr":           trial.suggest_categorical("lr", g["lr"]),
        "wd":           trial.suggest_categorical("wd", [float(w) for w in g["wd"]]),

        # feature engineering
        "ma_window":    trial.suggest_categorical("ma_window", g["ma_window"]),

        # dynamic TP / SL band construction
        "tp_sl_lookback_window":         trial.suggest_categorical("tp_sl_lookback_window", g["tp_sl_lookback_window"]),
        "min_range_pct":   trial.suggest_categorical("min_range_pct", g["min_range_pct"]),
    }

    # ─────────────────────────── 2) FEATURE BUILDING ───────────────────────────────
    # Grab MA‑window‑specific feature set and align it to TRAIN / TEST indices.

    # 2️⃣ build/cached features for this symbol & window
    feat_full = FEAT_CACHE[sym][hp["ma_window"]].drop_duplicates(keep="last")

    # 3️⃣ align features to each segment, drop rows missing features or duplicated indices
    df_tr_f = (
        feat_full
        .reindex(raw_train.index)
        .dropna()
        .loc[lambda d: ~d.index.duplicated(keep="last")]
    )
    df_te_f = (
        feat_full
        .reindex(raw_test.index)
        .dropna()
        .loc[lambda d: ~d.index.duplicated(keep="last")]
    )
    df_fu_f = (
        feat_full
        .reindex(raw_future.index)
        .dropna()
        .loc[lambda d: ~d.index.duplicated(keep="last")]
    )

    seeds   = [np.random.randint(500), np.random.randint(500), np.random.randint(500)]
    models, val_stacks = [], []        # ← no more isos
    for seed in seeds:
        torch.manual_seed(seed)
        try:
            mdl_i, preds_val_i = train_one(
                df_tr_f, sym, tf, hp, FEATS_LIST, device
            )
        except Exception as ex:
            log.info(f'Exception while training: {ex}')
            traceback.print_exc()
            return -1

        models.append(mdl_i)

        log.info(f'test')

        # convert validation logits → probabilities
        val_probs_i = 1 / (1 + np.exp(-preds_val_i))
        val_stacks.append(val_probs_i)

    # ── ensemble distribution on validation ───────────────────────────
    val_probs_stack = np.vstack(val_stacks)          # (n_models, n_val)
    ens_val_probs   = val_probs_stack.mean(axis=0)

    p10, p90 = np.percentile(ens_val_probs, [10, 90])
    min_th   = max(0.0, p10 - 0.05)
    max_th   = min(1.0, p90 + 0.05)
    if max_th - min_th < 0.05:                       # safety fallback
        min_th, max_th = 0.20, 0.80

    steps = max(10, int((int((max_th - min_th) * 100))/3))    # ~1‑pp grid
    log.info(f"[θ‑cal] search window: {min_th:.3f} – {max_th:.3f}  "
            f"({steps} steps)")

    # ── grid‑search θ on train / test / future slices ─────────────────
    th_tr, score_tr = calibrate_threshold(
        df_tr_f, models, hp, sym, tf, FEATS_LIST, device,
        min_th=min_th, max_th=max_th, steps=steps
    )
    log.info(f"train_th: {th_tr:.4f} → score: {score_tr:.4f}")

    th_te, score_te = calibrate_threshold(
        df_te_f, models, hp, sym, tf, FEATS_LIST, device,
        min_th=min_th, max_th=max_th, steps=steps
    )
    log.info(f"test_th:  {th_te:.4f} → score: {score_te:.4f}")

    th_fu, score_fu = calibrate_threshold(
        df_fu_f, models, hp, sym, tf, FEATS_LIST, device,
        min_th=min_th, max_th=max_th, steps=steps
    )
    log.info(f"fut_th:   {th_fu:.4f} → score: {score_fu:.4f}")

    best_th    = float(np.mean([th_tr, th_te, th_fu]))
    best_score = float(np.mean([score_tr, score_te, score_fu]))
    log.info(f"mean_th:  {best_th:.4f} → mean_score: {best_score:.4f}")

    hp["threshold"] = best_th

    # if score < 0:
    #     return -1

    # log.info(f"Train Score: {score}")

    # ─────────────────────────── 4) AUXILIARY HELPERS ─────────────────────────────
    # Soft logistic penalty (re‑used in several places)
    def soft_penalty(miss: float, tol: float, steep: float = 8.0) -> float:
        """
        Sigmoid‑shaped penalty in [0,1].
        miss  : positive = amount by which constraint is missed
        tol   : tolerated miss before penalty ramps up
        """
        return 1.0 / (1.0 + math.exp(-steep * (miss - tol)))

    # Permutation‑importance penalty
    def permute_and_score(
    df,
    feats,
    models,
    hp,
    base_sharpe,
    tf,
    device,
    target_drop=None,
    n_perm: int = 3,
    n_jobs: int = 4,
    verbose: bool = True,          # ← new flag
    ):
        """
        • Shuffle each feature n_perm times
        • Measure Sharpe deterioration
        • Return penalty multiplier + observed max drop

        If verbose=True, also prints per-feature avg Sharpe drop.
        """
        if target_drop is None:
            target_drop = CFG["checks"]["perm_delta"]

        # ---------- inner helper: shuffle single feature ----------
        def _drop_for(feat_name: str) -> float:
            drops = []
            for _ in range(n_perm):
                df_p = df.copy()
                df_p[feat_name] = np.random.permutation(df_p[feat_name].values)

                # Backtest on permuted data
                sub_p, ratios, sig_p = infer_and_threshold(
                    df_p, best_th, models, hp, sym, tf, FEATS_LIST, device
                )
                trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(
                    sub_p, sig_p, hp
                )
                sharpe_p = stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars)["sharpe"]
                drops.append(max(0.0, base_sharpe - sharpe_p))
            return float(np.mean(drops))

        # ---------- run shuffles in parallel ----------
        avg_drops = {}
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            future_map = {exe.submit(_drop_for, f): f for f in feats}
            for fut in as_completed(future_map):
                feat = future_map[fut]
                avg_drops[feat] = fut.result()

        obs_drop = float(np.clip(max(avg_drops.values(), default=0.0), 0.0, 2.0))
        log.info(f"Observed avg Sharpe drop (perm-imp): {obs_drop:.3f}")

        # ← new printing block
        if verbose:
            log.info("Feature-wise average Sharpe drops:")
            # sort descending by drop
            for feat, drop in sorted(avg_drops.items(), key=lambda kv: kv[1], reverse=True):
                log.info(f"  {feat:>10s}  →  {drop:.4f}")

        miss     = max(0.0, target_drop - obs_drop)
        pen_perm = soft_penalty(miss, tol=0.0, steep=10.0)

        # you could also return avg_drops if you want to inspect them downstream:
        return pen_perm, obs_drop
    
    def aggregate_slices(slices: dict[str, dict]) -> float:
        """
        slices is a dict like:
        {
            "TRAIN":  {"score": float, ...},
            "TEST":   {"score": float, ...},
            "FUTURE": {"score": float, ...},
        }
        We weight them 15/35/50%, but return 0.0 if any of the three scores is negative.
        """
        # extract the three scores
        t, v, f = slices["TRAIN"]["score"], slices["TEST"]["score"], slices["FUTURE"]["score"]
        
        # if any slice score is negative, zero out the aggregate
        if t <= 0 or v <= 0 or f <= 0:
            return 0.0
        
        # otherwise do the weighted sum
        return 0.50 * v + 0.50 * f

    # ─────────────────────────── 5) BACKTEST TRAIN / TEST SLICES ─────────────
    # ---------- TRAIN ----------
    sub_tr, ratios_tr, sig_tr  = infer_and_threshold(df_tr_f, best_th, models, hp, sym, tf, FEATS_LIST, device, debug=True)
    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(sub_tr, sig_tr, hp, debug=True)
    log.info("Train Scores")
    slices = {
        "TRAIN": {
            "stats": stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
            "score": slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
        }
    }

    if slices["TRAIN"]["score"] <= 0:
        return -1

    # ---------- TEST ----------
    log.info("Test Scores")
    sub_te, ratios_te, sig_te = infer_and_threshold(df_te_f, best_th, models, hp, sym, tf, FEATS_LIST, device, debug=True)
    n_te      = len(sub_te)

    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(sub_te, sig_te, hp, debug=True)
    slices["TEST"] = {
        "stats": stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
        "score": slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
    }

    if slices["TEST"]["score"] <= 0:
        return -1

    # ─────────────────────────── 6) OPTIONAL “FUTURE” SLICE ──────────────────
    sub_fu, ratios_fu, sig_fu = infer_and_threshold(df_fu_f, best_th, models, hp, sym, tf, FEATS_LIST, device, debug=True)

    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(sub_fu, sig_fu, hp, debug=True)
    log.info("Future Scores")
    slices["FUTURE"] = {
        "stats": stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
        "score": slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
    }

    if slices["FUTURE"]["score"] <= 0:
        return -1

    # ─────────────────────────── 7) RANDOM BASELINE (Monte‑Carlo) ────────────
    MC          = 5
    np.random.seed(42)

    n_model_trades = int(sig_te.sum())          # match #trades exactly
    rand_scores    = []

    for _ in range(MC):
        idx           = np.random.choice(n_te, size=n_model_trades, replace=True)
        sig_rand      = np.zeros(n_te, dtype=bool)
        sig_rand[idx] = True

        trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(sub_te, sig_rand, hp)
        rand_scores.append(
            slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug=False)
        )

    if not rand_scores:
        mu_rand, sigma_rand = 0.0, 0.05        # benign defaults  (5 % band)
    else:
        mu_rand, sigma_rand = np.mean(rand_scores), np.std(rand_scores, ddof=1)

    sigma_floor = max(0.01, 0.10 * abs(mu_rand))   # ≥1 % of |μ|
    if sigma_rand < sigma_floor:
        log.warning(f"Low σ ({sigma_rand:.4f}) → flooring to {sigma_floor:.4f}")
        sigma_rand = sigma_floor

    # ─────────────────────────── 8) MODEL RAW GAIN (Z‑score) ────────────────
    aggregate_score = aggregate_slices(slices)

    log.info(f"Aggregated Score = {aggregate_score:.5f}")

    z_score  = (aggregate_score - mu_rand) / sigma_rand
    z_score  = np.clip(z_score, -2, +2)            # optional hard cap

    # keep sign of original score
    if z_score > 0:
        raw_gain = z_score * (-1 if aggregate_score < 0 else 1)
    else:
        raw_gain = z_score


    log.info(f"Model slice score   = {aggregate_score:.4f}")
    log.info(f"Random50 (μ, σ)     = ({mu_rand:.4f}, {sigma_rand:.4f})")
    log.info(f"Model Z-score       = {z_score:.2f}")

    raw_gain = raw_gain * abs(aggregate_score)

    log.info(f"Raw Gain       = {raw_gain:.5f}")

    # Permutation‑importance
    base_sharpe        = slices["TEST"]["stats"]["sharpe"]
    pen_perm, obs_drop = permute_and_score(df_te_f, FEATS_LIST, models, hp, base_sharpe,
                                           tf, device,
                                           target_drop=CFG["checks"]["perm_delta"],
                                           n_perm=2, n_jobs=1)
    bonus_perm = 1.0 + 0.50 * max(0.0, obs_drop)
    m_perm     = max(0.05, (1.0 - 0.50 * pen_perm) * bonus_perm)

    # ─────────────────────────── 10) FINAL HEADLINE SCORE ───────────────────
    score = raw_gain * m_perm
    log.info(
        f"raw_gain={raw_gain:.4f} " 
        f"m_perm={m_perm:.3f} "
        f"→ headline={score:.4f}"
    )

    # ─────────────────────────── 11) PERSISTENCE / BOOKKEEPING ──────────────
    tag = f"{sym.replace('/', '')}_{tf.value.replace('/', '')}"
    details = {nm: {**v["stats"], "Score_slice": v["score"]} for nm, v in slices.items()}
    details["Aggregate"] = {"Score": score, "HyperParams": hp}

    # ---- 1) bundle up ensemble components into dicts ----
    ensemble_last = []
    ensemble_best = []

    for idx, mdl_i in enumerate(models):
        checkpoint = {
            "state_dict": mdl_i.state_dict(),
            "params":     hp
        }
        ensemble_last.append(checkpoint)

        # if this is the new best, prepare the same for best_ensemble
        if score > BEST_PER_SERIES.get(tag, -np.inf):
            ensemble_best.append(checkpoint)

    # ---- 2) write “last” ensemble to disk ----
    last_path = TRAINED_MODELS / f"last_{tag}_ensemble.pkl"
    with open(last_path, "wb") as f:
        pickle.dump(ensemble_last, f)
    PARAMS_DIR.joinpath(f"last_params_{tag}.json")\
        .write_text(json.dumps(details, indent=2))

    # ---- 3) update & write “best” ensemble if needed ----
    log.info(f"Current Symbol: {sym}")
    log.info(f"Current Best Score: {BEST_PER_SERIES.get(tag, '–')}")
    log.info(f"Current Score: {score}")

    if score > BEST_PER_SERIES.get(tag, -np.inf):
        BEST_PER_SERIES[tag] = score

        best_path = TRAINED_MODELS / f"best_{tag}_ensemble.pkl"
        with open(best_path, "wb") as f:
            pickle.dump(ensemble_last, f)   # reuse ensemble_last since it's the same
        PARAMS_DIR.joinpath(f"best_params_{tag}.json")\
            .write_text(json.dumps(details, indent=2))
        log.info(f"[{tag}] new BEST score {score:.5f}")

    # ─────────────────────────── 12) RETURN OBJECTIVE VALUE ─────────────────
    return score

# ─────────────────────────────── main loop ─────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device → {device}")

    syms   = CFG["symbols"]
    tf     = TimeFrame(30, TimeFrameUnit.Minute)
    key_tf = tf.value

    # 1) fetch raw bars in parallel
    raws: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(1, os.cpu_count())) as ex:
        futs = {}
        for sym in syms:
            cache_key = (sym, key_tf)
            if cache_key in RAW_CACHE:
                raws[sym] = RAW_CACHE[cache_key]
            else:
                futs[ex.submit(fetch, sym, tf, CFG["start_date"], CFG["end_date"])] = sym

        for fut in as_completed(futs):
            sym = futs[fut]
            raw = fut.result()
            raw = raw[~raw.index.duplicated(keep="last")]
            RAW_CACHE[(sym, key_tf)] = raw
            raws[sym] = raw

    common_idx = None
    for df in raws.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    # **NEW: save it for later**
    pd.to_pickle(common_idx, "common_idx.pkl")

    for sym in raws:
        raws[sym] = raws[sym].reindex(common_idx)

    # ── 2) single‑pass optimisation per symbol – no walk‑forward windows ─────────
    for sym, raw in raws.items():
        if raw.empty:
            continue                                    # nothing to optimise

        raw = raw.sort_index()                          # just in case

        # ── 1. build / cache features for all ma windows ─────────────────────
        for mw in CFG["grid"]["ma_window"]:
            if mw not in FEAT_CACHE[sym]:
                FEAT_CACHE[sym][mw] = build_feats(raw, mw, sym)

        n = len(raw)
        train_end = int(n * 0.7)
        test_end  = int(n * 0.9)

        # 1️⃣ slice raw into three segments
        raw_train  = raw.iloc[:train_end]
        raw_test   = raw.iloc[train_end:test_end]
        raw_future = raw.iloc[test_end:]

        # 4️⃣ create your Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        # 5️⃣ wrap the objective so it sees all three splits
        def wrapped_obj(trial):
            return float(
                objective(
                    trial,
                    sym,
                    tf,         # timeframe
                    raw_train,    # 70% train
                    raw_test,    # 20% test
                    raw_future,    # 10% future
                    device,
                )
            )

        study.optimize(
            wrapped_obj,
            n_trials=CFG["n_trials"],
            n_jobs=min(1, os.cpu_count()),
        )

if __name__ == "__main__":
    main()