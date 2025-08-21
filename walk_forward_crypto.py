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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle
import traceback
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import yaml, optuna
from optuna.samplers import TPESampler
from torch.utils.data import SubsetRandomSampler

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, Sort
from alpaca.data.enums import CryptoFeed
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view   # already vectorised
import copy

import itertools

from binance.client import Client


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

BINANCE_API_KEY = CFG["binance_api_key"]
BINANCE_API_SECRET = CFG["binance_api_secret"]

binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld="us")

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

def fetch_binance(sym: str, interval: str, start, end,
                  max_retries=5, base_sleep=1.5) -> pd.DataFrame:
    """
    Fetch *all* historical bars from Binance Spot for symbol `sym` at `interval`
    between `start` and `end`, with extra fields, and verbose logging.
    """
    log.info(f"Starting fetch_binance for {sym} {interval} from {start} to {end}")

    # convert to UTC-ms
    start_ts = int(pd.to_datetime(start).tz_localize("UTC").timestamp() * 1000)
    end_ts   = int(pd.to_datetime(end).tz_localize("UTC").timestamp() * 1000)

    cache_file = CACHE_DIR / f"{sym.replace('/','')}_{interval}_{start_ts}_{end_ts}.pkl"
    if cache_file.exists():
        log.info(f"Loading from cache: {cache_file.name}")
        return pd.read_pickle(cache_file)

    # how many ms per bar?
    interval_ms = int(pd.to_timedelta(interval).total_seconds() * 1000)
    all_bars = []
    next_start = start_ts
    chunk_count = 0

    while True:
        log.info(f"Requesting chunk #{chunk_count+1}: startTime={next_start}, endTime={end_ts}")
        # retry loop for this chunk
        for attempt in range(1, max_retries + 1):
            try:
                chunk = binance.get_klines(
                    symbol=sym.replace("/", ""),
                    interval=interval,
                    startTime=next_start,
                    endTime=end_ts,
                    limit=1000
                )
                log.info(f"  → Received {len(chunk)} bars on attempt {attempt}")
                break
            except HTTPError as err:
                code = err.response.status_code
                if code == 429 and attempt < max_retries:
                    wait = base_sleep * 2 ** (attempt - 1) + random.random() * base_sleep
                    log.warning(f"  → Rate limited (429) on attempt {attempt}, retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    log.error(f"  → HTTP error {code} on attempt {attempt}, giving up chunk")
                    raise

        if not chunk:
            log.info("No more bars returned; ending loop")
            break

        all_bars.extend(chunk)
        chunk_count += 1

        last_open = chunk[-1][0]  # open_time in ms
        next_start = last_open + interval_ms
        log.info(f"  → Next startTime set to {next_start} (last_open + interval)")

        if next_start >= end_ts:
            log.info("Reached end timestamp; ending loop")
            break

        # avoid hammering API
        time.sleep(0.1)

    log.info(f"Fetched total {len(all_bars)} bars across {chunk_count} chunks; assembling DataFrame")

    # unpack into DataFrame
    df = pd.DataFrame(all_bars, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ]).astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "quote_volume": float,
        "taker_base_vol": float, "taker_quote_vol": float,
        "trade_count": int
    })

    df["vwap"] = df["quote_volume"] / df["volume"]
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[[
        "open","high","low","close",
        "volume","vwap","trade_count",
        "quote_volume","taker_base_vol","taker_quote_vol"
    ]]

    log.info(f"Caching DataFrame to {cache_file.name}")
    df.to_pickle(cache_file)

    log.info("fetch_binance complete\n")
    return df

# ───────────────────────── feature builder ───────────────────────────────

def build_feats(df: pd.DataFrame, ma: int, sym: str) -> pd.DataFrame:
    if ma <= 0:
        raise ValueError("ma must be positive")
    d = df.copy()

    # ─── Price‐based momentum & trend ───────────────────────────
    d['ma']    = d['close'].rolling(ma).mean()
    d['mom']   = d['ma'].diff()
    d['force'] = d['mom'].diff()
    d['jerk']  = d['force'].diff()

    # RSI
    delta = d['close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    d['rsi'] = 100 - 100 / (1 + gain.rolling(14).mean() / (loss.rolling(14).mean() + EPS))

    # ATR
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - d['close'].shift()).abs(),
        (d['low']  - d['close'].shift()).abs()
    ], axis=1).max(axis=1)
    d['atr'] = tr.rolling(14).mean()

    # Returns & EMA
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
    d['boll_b'] = (d['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + EPS)
    d['boll_w'] = (4*bb_std) / (bb_mid + EPS)

    # Stochastic Oscillator
    low14  = d['low'].rolling(14).min()
    high14 = d['high'].rolling(14).max()
    d['stoch_k'] = 100*(d['close']-low14)/(high14-low14+EPS)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()


    # ─── Volume & Order‐flow ───────────────────────────────────
    # normalize base‐volume
    d['vol_rel'] = d['volume'] / (d['volume'].rolling(20).mean() + EPS)
    # quote‐volume analog
    d['qvol_ma']   = d['quote_volume'].rolling(20).mean()
    d['qvol_rel']  = d['quote_volume'] / (d['qvol_ma'] + EPS)
    d['qvol_mom']  = d['qvol_ma'].diff()

    # on‐balance volume (base)
    d['obv'] = (np.sign(d['close'].diff()) * d['volume']).fillna(0).cumsum()

    # taker buy / sell features
    d['taker_buy_ratio']     = d['taker_base_vol'] / (d['volume'] + EPS)
    d['taker_buy_imbalance'] = (d['taker_base_vol'] - (d['volume'] - d['taker_base_vol'])) \
                               / (d['volume'] + EPS)
    d['taker_quote_ratio']   = d['taker_quote_vol'] / (d['quote_volume'] + EPS)

    # average trade‐size
    d['avg_trade_base']   = d['volume'] / (d['trade_count'] + EPS)
    d['avg_trade_quote']  = d['quote_volume'] / (d['trade_count'] + EPS)
    d['trade_count_rel']  = d['trade_count'] / (d['trade_count'].rolling(20).mean() + EPS)

    # realized volatility (20-bar)
    d['rv_20'] = d['ret1'].rolling(20).std()

    # high-low range
    d['hl_range'] = (d['high'] - d['low']) / (d['low'] + EPS)


    # ─── VWAP deviations & session VWAP ────────────────────────
    d['vwap_dev'] = (d['close'] - d['vwap']) / (d['vwap'] + EPS)

    # 1-day session VWAP (48×30m bars)
    window = 48
    num = (d['close'] * d['volume']).rolling(window).sum()
    den = d['volume'].rolling(window).sum() + EPS
    d['session_vwap']     = num / den
    d['session_vwap_dev'] = (d['close'] - d['session_vwap']) / (d['session_vwap'] + EPS)


    # # ─── Time features ────────────────────────────────────────
    # # time‐of‐day cycle
    # t = d.index.tz_convert(None)
    # mins = t.hour*60 + t.minute
    # d['tod_sin'] = np.sin(2*np.pi*mins/1440)
    # d['tod_cos'] = np.cos(2*np.pi*mins/1440)

    # # day‐of‐week cycle
    # dow = t.dayofweek
    # d['dow_sin'] = np.sin(2*np.pi * dow / 7)
    # d['dow_cos'] = np.cos(2*np.pi * dow / 7)


    # fill any remaining NaNs and return
    return d.fillna(0)


FEATS_LIST = [
    "open", "high", "low", "close", "volume", "vwap", "trade_count", "quote_volume", "ma","mom","force","jerk",
    "rsi","atr",
    "ret1",
    "macd","macd_sig",
    "boll_b","boll_w",
    "stoch_k","stoch_d","qvol_mom","obv",
    "avg_trade_quote","trade_count_rel",
    "rv_20","hl_range",
    "vwap_dev","session_vwap","session_vwap_dev", "taker_buy_imbalance", "taker_buy_ratio","taker_quote_ratio", "avg_trade_base", "vol_rel",  "ret_ema", "qvol_rel"]
    #"tod_sin","tod_cos","dow_sin","dow_cos", 

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FEE_PCT        = 0.00           # fraction of trade value charged both sides
SIM_INIT_CAP   = 1_000_000      # starting capital in account currency
PCT_PER_TRADE  = 0.40           # Kelly‑fraction cap (position sizing)
DTYPE_F32      = np.float32

# -----------------------------------------------------------------------------
# SimpleBacktester – second revision (adds edge‑z filter & tuple return‑format)
# -----------------------------------------------------------------------------
class SimpleBacktester:
    """
    Long‑only back‑tester implementing §§ 3.2 – 3.4 of the trading plan.

    Entry logic now follows the **edge‑z rule** described by the user:
        edge_z = Δ / u
        enter long if edge_z ≥ λ  and  Δ > 0
    where λ comes from the *entry_k* hyper‑parameter (default 1.0).

    Exit rules use the reward band R = (L/H)·Δ as specified:
        TP = MAₜ + k_tp · R
        SL = P₀  − k_sl · R
        time‑exit at bar t + H

    Shorts are explicitly disabled.

    The *run* method now returns the statistics **exactly** in the tuple order
    requested by the user.
    """

    def __init__(
        self,
        *,
        hp: dict,
        initial_balance: float = SIM_INIT_CAP,
        debug: bool = False,
    ):
        # --- hyper‑parameters -----------------------------------------------
        self.horizon    = int(hp["horizon"])             # H
        self.ma_window  = int(hp["ma_window"])           # L (≥ H)
        self.take_k     = float(hp.get("take_k", 0.8))   # k_tp
        self.stop_k     = float(hp.get("stop_k", 1.0))   # k_sl
        self.edge_lambda = float(hp.get("entry_k", 1.0)) # λ (edge threshold)
        self.fee_rate   = float(hp.get("fee_rate", FEE_PCT))

        self.initial_balance = float(initial_balance)
        self.debug     = debug
        self._reset()

    # ---------------------------------------------------------------- helpers
    def _reset(self):
        self.balance = self.initial_balance   # current cash balance
        self.pnl     = 0.0                    # cumulative profit / loss
        self.opened  = []                     # list of open positions
        self.closed  = []                     # list of dictionaries for closed positions

    def _close(self, pos: dict, price: float, idx_exit: int, exit_type: str):
        """Close *pos* at *price* on bar *idx_exit* and register statistics."""

        qty        = pos["qty"]
        fee_exit   = self.fee_rate * price * qty
        gross_pnl  = (price - pos["entry_price"]) * qty
        net_pnl    = gross_pnl - pos["fee_entry"] - fee_exit

        pos.update(
            exit_idx   = idx_exit,
            exit_price = price,
            fee_exit   = fee_exit,
            gross_pnl  = gross_pnl,
            net_pnl    = net_pnl,
            hold_bars  = idx_exit - pos["entry_idx"],
            exit_type  = exit_type,
        )

        self.balance += qty * price - fee_exit
        self.pnl     += net_pnl
        self.closed.append(pos)

    # ---------------------------------------------------------------- run
    def run(self, aligned_df: pd.DataFrame, signals: np.ndarray):
        """Execute the back‑test over *aligned_df* using *signals* (0/1)."""

        self._reset()

        H, L = self.horizon, self.ma_window
        if L < H:
            raise ValueError("ma_window (L) must be ≥ horizon (H)")
        r_LH = L / H                          # r = L / H

        # --- pre‑compute series ---------------------------------------------
        close_s = aligned_df["close"].ffill().bfill()
        ma_s    = close_s.rolling(window=L, min_periods=L).mean()

        
        perfect_delta_series = ma_s - ma_s.shift(H)
        perfect_deltas = perfect_delta_series.to_numpy(dtype=DTYPE_F32)

        closes  = close_s.to_numpy(dtype=DTYPE_F32)
        mas     = ma_s.to_numpy(dtype=DTYPE_F32)
        deltas  = perfect_deltas # aligned_df["delta_pred"].to_numpy(dtype=DTYPE_F32)
        uncs    = aligned_df["unc"].to_numpy(dtype=DTYPE_F32)

        n = len(aligned_df)

        # --------------------------------------------------------------------
        for idx in range(n):
            bar_high = aligned_df["high"].iloc[idx]
            bar_low  = aligned_df["low"].iloc[idx]

            if idx > 500 and idx < 550:
                print(f'Idx: {idx}')
                print(f'Delta: {deltas[idx]}')
                print(f'Actual Delta: {perfect_deltas[idx]}')

            # ─── 1. update open trades ───────────────────────────────────────
            for pos in self.opened[:]:
                if idx <= pos["entry_idx"]:
                    continue  # cannot exit in the same bar we entered

                # stop‑loss first (conservative)
                if bar_low <= pos["sl_price"]:
                    self._close(pos, pos["sl_price"], idx, "stop")
                    self.opened.remove(pos)
                    continue

                # take‑profit
                if bar_high >= pos["tp_price"]:
                    self._close(pos, pos["tp_price"], idx, "target")
                    self.opened.remove(pos)
                    continue

                # horizon exit (time stop)
                if idx >= pos["exit_idx"]:
                    self._close(pos, closes[idx], idx, "horizon")
                    self.opened.remove(pos)

            # ─── 2. new entry? ───────────────────────────────────────────────
            if idx >= n - H - 1:
                continue  # not enough bars left for full horizon
            if signals[idx] != 1 or not np.isfinite(mas[idx]):
                continue

            # --- edge‑z filter ----------------------------------------------
            d_hat = float(deltas[idx])
            u_hat = float(uncs[idx]) if uncs[idx] > 0 else np.nan
            if np.isnan(u_hat):
                continue  # cannot evaluate edge

            edge_z = d_hat / u_hat
            if d_hat <= 0 or edge_z < self.edge_lambda:
                continue  # no shorts & must beat λ sigma edge

            # --- position sizing & bookkeeping -----------------------------
            entry_price = closes[idx]
            qty = (PCT_PER_TRADE * self.balance) / entry_price
            if qty <= 0:
                continue

            fee_entry = self.fee_rate * entry_price * qty
            self.balance -= qty * entry_price + fee_entry

            # --- derive reward band & levels --------------------------------
            R  = r_LH * d_hat                     # reward band (price units)
            tp_price = mas[idx] + self.take_k * R # MAₜ + k_tp·R
            sl_price = entry_price - self.stop_k * R

            # Optional guard: ensure TP above entry (avoids pathological cases)
            if tp_price <= entry_price:
                tp_price = entry_price * 1.0001   # ≈ +1 bp

            self.opened.append({
                "entry_idx"   : idx,
                "entry_price" : entry_price,
                "qty"         : qty,
                "fee_entry"   : fee_entry,
                "exit_idx"    : idx + H,        # horizon bar index
                "tp_price"    : tp_price,
                "sl_price"    : sl_price,
            })

        # ─── 3. forcibly close leftovers on last bar ─────────────────────────
        if self.opened:
            last_idx   = n - 1
            last_close = closes[last_idx]
            for pos in self.opened:
                self._close(pos, last_close, last_idx, "horizon")
            self.opened.clear()

        # ─── 4. statistics ───────────────────────────────────────────────────
        trades = len(self.closed) or 1  # keep non‑zero to avoid NaNs
        trade_rate = trades / n

        # classify exits
        exit_counter  = Counter(tr["exit_type"] for tr in self.closed)
        exit_target   = exit_counter.get("target", 0)
        exit_stop     = exit_counter.get("stop", 0)
        exit_horizon  = exit_counter.get("horizon", 0)

        # performance numbers
        winners   = [tr["net_pnl"] for tr in self.closed if tr["net_pnl"] > 0]
        losers    = [tr["net_pnl"] for tr in self.closed if tr["net_pnl"] <= 0]
        win_rate  = 100 * len(winners) / trades

        # R‑multiples expressed as fraction of initial capital (simple proxy)
        avg_R_win  = np.mean(winners) / self.initial_balance if winners else 0.0
        avg_R_loss = np.mean(losers)  / self.initial_balance if losers  else 0.0
        expectancy_R = (win_rate/100) * avg_R_win + (1 - win_rate/100) * avg_R_loss

        pct_returns = [(tr["net_pnl"] / (tr["entry_price"] * tr["qty"])) for tr in self.closed]
        sharpe = (np.mean(pct_returns) / np.std(pct_returns) * np.sqrt(trades)
                  if trades >= 2 and np.std(pct_returns) > 0 else 0.0)

        avg_hold_bars = float(np.mean([tr["hold_bars"] for tr in self.closed]))
        pct_gain      = self.pnl / self.initial_balance

        # optional debug printout -------------------------------------------
        if self.debug:
            print("# --- Back‑test stats --------------------------------------------------")
            print(f"Trades            : {trades}")
            print(f"Exit‑via target   : {100 * exit_target  / trades:6.2f}%")
            print(f"Exit‑via stop     : {100 * exit_stop    / trades:6.2f}%")
            print(f"Exit‑via horizon  : {100 * exit_horizon / trades:6.2f}%")
            print(f"Win rate          : {win_rate:6.2f}%")
            print(f"Avg R winners     : {avg_R_win: .5f}")
            print(f"Avg R losers      : {avg_R_loss: .5f}")
            print(f"Expectancy R      : {expectancy_R: .5f}")
            print(f"Sharpe            : {sharpe: .2f}")
            print(f"Ending balance    : {self.balance:,.2f}")
            print(f"PNL               : {self.pnl:,.2f}")
            print(f"Percent gains     : {pct_gain: .2%}")
            print(f"Avg hold bars     : {avg_hold_bars: .2f}\n")

        # --- tuple output in the exact user‑requested order -----------------
        return (
            trades,
            trade_rate,
            win_rate,
            avg_R_win,
            avg_R_loss,
            expectancy_R,
            self.balance,
            self.pnl,
            pct_gain,
            sharpe,
            avg_hold_bars,
        )




# ────────────────────────── thin wrapper ────────────────────────────────────
def run_backtest(
    aligned_df : pd.DataFrame,
    signals  : np.ndarray,
    hp       : dict,
    *,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    bt = SimpleBacktester(hp=hp, initial_balance=SIM_INIT_CAP, debug=debug)

    return bt.run(aligned_df, signals)

###############################################################################
# 1.  Feature windows – rolling z‑score (look‑back = hp["rolling_size"])
###############################################################################
def prepare_windows(
    df: pd.DataFrame,
    feats: list[str],
    seq_len: int,
    rolling_size: int,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Returns X of shape (n_windows, seq_len, n_feats) where **each point** in the
    window is normalised with the *rolling* mean / std computed over the
    previous `rolling_size` rows (inclusive).  This prevents information from
    the *future* leaking into the normalisation statistics.

        x_norm[t] = (x[t] – μᵣ[t]) / (σᵣ[t] + eps)

    Any NaN/Inf is replaced with 0.
    """
    df_filled = df[feats].ffill().bfill()              # ensure no gaps
    x         = df_filled.values.astype(np.float32)    # (T, n_feats)

    # rolling μ and σ – uses past `rolling_size` rows (inclusive)
    mu_roll = (
        df_filled.rolling(window=rolling_size, min_periods=1)
        .mean()
        .values.astype(np.float32)
    )
    sd_roll = (
        df_filled.rolling(window=rolling_size, min_periods=1)
        .std(ddof=0)        # population σ; avoids tiny sample penalties
        .values.astype(np.float32)
        + eps
    )

    x_norm = (x - mu_roll) / sd_roll
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # finally carve the (seq_len, n_feats) windows out of the *normalised* array
    win = np.lib.stride_tricks.sliding_window_view(
        x_norm, window_shape=(seq_len, x_norm.shape[1])
    ).squeeze(1)                                        # (n_w, seq_len, n_feats)

    return np.ascontiguousarray(win)


###############################################################################
# 2.  Target vector – always Δ, always standardised
###############################################################################
def make_target_vector(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    ma_window: int,
) -> Tuple[np.ndarray, float, float]:
    """
    y_raw[t] = MA_{t+H} − MA_t             (Δ change).
    Returned y is z‑scored with global μ, σ computed on *training* split.
    """
    close = df["close"].ffill().bfill().values.astype(np.float32)

    ma = (
        pd.Series(close)
        .rolling(window=ma_window, min_periods=1)
        .mean()
        .values.astype(np.float32)
    )

    delta = ma[horizon:] - ma[:-horizon]        # length T‑H
    y_raw = delta[seq_len - 1 :]                # align to feature windows

    mu, sd = float(y_raw.mean()), float(y_raw.std() + 1e-8)
    y = (y_raw - mu) / sd
    return y.astype(np.float32), mu, sd


###############################################################################
# 3.  Dataset – passes `rolling_size` into prepare_windows
###############################################################################
class SequenceDataset(Dataset):
    """
    Produces (window, y_normed).  Attributes `y_mu`, `y_sd` for inverse‑transform.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feats: list[str],
        seq_len: int,
        horizon: int,
        ma_window: int,
        rolling_size: int,          # NEW
    ):
        self.X = prepare_windows(
            df, feats, seq_len, rolling_size
        )
        self.y, self.y_mu, self.y_sd = make_target_vector(
            df, seq_len, horizon, ma_window
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


###############################################################################
# 3.  Model – same backbone, linear head with *no* sigmoid
###############################################################################
class MAValueModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)   # regression output

    def forward(self, x):
        _, h = self.rnn(x)                      # h: (num_layers, B, H)
        if isinstance(h, tuple):                # LSTM returns (h_n, c_n)
            h = h[0]
        return self.head(h[-1]).squeeze(-1)     # (B,)


###############################################################################
# 4.  Regression metrics helper
###############################################################################
def _reg_metrics(pred_list, label_list) -> dict:
    """
    Compute MAE, RMSE and R² on two lists of tensors.
    """
    preds  = torch.cat(pred_list).cpu().numpy()
    labels = torch.cat(label_list).cpu().numpy()

    mae  = np.abs(preds - labels).mean()
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    ss_tot = np.var(labels) * len(labels)
    r2   = 1.0 - np.sum((preds - labels) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return dict(mae=mae, rmse=rmse, r2=r2)


###############################################################################
# 5.  Single epoch (train or eval) – uses SmoothL1 loss
###############################################################################
def _epoch(dataloader, model, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_samples = 0.0, 0
    preds_all, labels_all = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss  = criterion(preds, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs
        preds_all.append(preds.detach())
        labels_all.append(y.detach())

    metrics  = _reg_metrics(preds_all, labels_all)
    avg_loss = total_loss / n_samples
    return avg_loss, metrics           # metrics = dict(mae, rmse, r2)


###############################################################################
# 6.  Primary training routine for regression
###############################################################################
def train_one(
    df_train: pd.DataFrame,
    hp: Dict[str, Any],
    feats: List[str],
    device: torch.device,
):
    """
    Trains the MA‑Δ regressor using fixed normalisation rules.
    """
    # ---------------  temporal split (80 / 20)
    n_val = int(len(df_train) * 0.20)
    df_val = df_train.iloc[-n_val:]
    df_tr  = df_train.iloc[:-n_val]

    # ---------------  datasets/dataloaders
    train_ds = SequenceDataset(
        df_tr,
        feats,
        hp["seq_len"],
        hp["horizon"],
        hp["ma_window"],
        hp["rolling_size"],        # NEW
    )
    val_ds = SequenceDataset(
        df_val,
        feats,
        hp["seq_len"],
        hp["horizon"],
        hp["ma_window"],
        hp["rolling_size"],        # NEW
    )

    train_dl = DataLoader(
        train_ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True
    )
    val_dl   = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False)

    # -------------------------------  model, loss, optimiser
    model = MAValueModel(
        len(feats), hp["hidden_size"], hp["num_layers"], hp["dropout"]
    ).to(device)

    criterion = nn.SmoothL1Loss()        # Huber – robust to tails
    optim     = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    # -------------------------------  training loop
    best_rmse, best_state = np.inf, None
    patience_left = hp["patience"]

    for ep in range(1, hp["max_epochs"] + 1):
        tr_loss, tr_m = _epoch(train_dl, model, criterion, optim, device)
        va_loss, va_m = _epoch(val_dl,   model, criterion, None,  device)

        log.info(
            "Ep %2d | "
            "train L %.5f  MAE %.4f  RMSE %.4f | "
            "val L %.5f  MAE %.4f  RMSE %.4f  R² %.3f | "
            "best %.4f  pat %d",
            ep,
            tr_loss, tr_m["mae"], tr_m["rmse"],
            va_loss, va_m["mae"], va_m["rmse"], va_m["r2"],
            best_rmse, patience_left,
        )

        # early‑stopping on RMSE
        if va_m["rmse"] < best_rmse:
            best_rmse, best_state = va_m["rmse"], model.state_dict()
            patience_left = hp["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info("Early‑stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log.warning("No RMSE improvement; returning last weights.")

    # -------------------------------  val predictions for inspection
    model.eval()
    preds_val = []
    with torch.no_grad():
        for x, _ in val_dl:
            preds_val.append(model(x.to(device)).cpu())
    preds_val = torch.cat(preds_val).numpy()

    # ------------------------------------------------------------------
    #  (Optional) feature‑importance logging, unchanged
    # ------------------------------------------------------------------
    try:
        w_ih = model.rnn.weight_ih_l0           # for GRU: weight_ih_l0
        imp_vec  = w_ih.abs().sum(dim=0)        # (input_dim,)
        feat_imp = sorted(zip(feats, imp_vec.cpu().tolist()),
                          key=lambda x: x[1],
                          reverse=True)
        log.info("Top feature importances:")
        for name, score in feat_imp[:10]:
            log.info("  %s: %.4f", name, score)
    except Exception as e:
        log.warning("Could not compute feature importances: %s", e)

    return model, preds_val



# ─────────────────── Optuna objective (single metric) ──────────────────────
BEST_PER_SERIES = defaultdict(lambda:-np.inf)
GLOBAL_TOP: List[Tuple[float,str,dict,dict]] = []  # (score, tag, sd, hp)


def slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug: bool = False) -> float:
    """Compute a composite score for a backtest slice, with detailed logging when debug=True."""

    final_score = percent_gains

    if trade_rate < .1:
         return -1

    return final_score

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Inference & signal generation  (long‑only)
# ─────────────────────────────────────────────────────────────────────────────
def infer_and_threshold(
    df: pd.DataFrame,
    threshold: float,                     # λ in § 3.1  (e.g. 1.2)
    models: list[torch.nn.Module],
    hp: dict,
    feats: list[str],
    device: torch.device,
    debug: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
        • aligned_df  – slice (seq_len … T−H) with "delta_pred" & "unc" columns
        • delta_preds – Δ̂  per row
        • signals     – 1 (long) | 0 (flat)  per row
    """
    seq_len   = int(hp["seq_len"])
    horizon   = int(hp["horizon"])
    ma_window = int(hp["ma_window"])
    rolling   = int(hp["rolling_size"])
    batch_sz  = int(hp.get("batch_size", 512))

    # same normalisation as training
    full_ds = SequenceDataset(df, feats, seq_len, horizon, ma_window, rolling)
    full_dl = DataLoader(full_ds, batch_size=batch_sz, shuffle=False)

    for m in models:
        m.to(device).eval()

    preds_per_model = [torch.zeros(len(full_ds)) for _ in models]
    with torch.no_grad():
        ofs = 0
        for xb, _ in full_dl:
            xb = xb.to(device)
            b  = xb.size(0)
            for i, m in enumerate(models):
                preds_per_model[i][ofs:ofs + b] = m(xb).cpu()
            ofs += b

    # destandardise
    y_mu, y_sd   = full_ds.y_mu, full_ds.y_sd
    stack        = torch.stack(preds_per_model)          # (M, T′)
    delta_preds  = (stack.mean(0) * y_sd + y_mu).numpy()
    unc          = (stack * y_sd).numpy().std(axis=0, ddof=0)

    λ       = float(threshold)
    signals = (delta_preds > λ * unc).astype(np.int8)    # 1 = long, else 0

    if debug:
        p10, p50, p90 = np.percentile(delta_preds, [10, 50, 90])
        print(f"Δ̂  p10={p10:.4f}, med={p50:.4f}, p90={p90:.4f}")
        print(f"Positive‑signal rate: {signals.mean()*100:.2f}%  (λ={λ})")

    start, end = seq_len - 1, len(df) - horizon
    aligned_df = df.iloc[start:end].copy()
    aligned_df["delta_pred"] = delta_preds
    aligned_df["unc"]        = unc

    return aligned_df, delta_preds, signals

# ─────────────────────────────┐
#  Revised calibrate_threshold ↓
# ─────────────────────────────┘
def calibrate_threshold(
    df: pd.DataFrame,
    models: list[torch.nn.Module],
    hp: dict,
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
        aligned_df, probs, sigs = infer_and_threshold(
            df, θ, models, hp, feats, device, debug=debug
        )
        trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df, sigs, hp, debug=debug)
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
        "seq_len":       trial.suggest_categorical("seq_len", g["seq_len"]),
        "horizon":       trial.suggest_categorical("horizon", g["horizon"]),
        "num_layers":    trial.suggest_categorical("num_layers", g["num_layers"]),
        "hidden_size":   trial.suggest_categorical("hidden_size", g["hidden_size"]),
        "dropout":       trial.suggest_categorical("dropout", g["dropout"]),

        # training / optimiser
        "max_epochs":        trial.suggest_categorical("max_epochs", g["max_epochs"]),
        "lr":            trial.suggest_categorical("lr", g["lr"]),
        "weight_decay":  trial.suggest_categorical("weight_decay", [float(w) for w in g["weight_decay"]]),
        "batch_size":    trial.suggest_categorical("batch_size", g["batch_size"]),
        "patience":      trial.suggest_categorical("patience", g.get("patience", [12])),

        # feature engineering
        "ma_window":     trial.suggest_categorical("ma_window", g["ma_window"]),
        "take_k":     trial.suggest_float("take_k", min(g["take_k"]), max(g["take_k"])),
        "rolling_size":     trial.suggest_categorical("rolling_size", g["rolling_size"]),
        "stop_k":     trial.suggest_float("stop_k", min(g["stop_k"]), max(g["stop_k"])),
        "entry_k":     trial.suggest_float("entry_k", min(g["entry_k"]), max(g["entry_k"])),
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
    models, val_stacks = [], []       # ← no more isos
    for seed in seeds:
        torch.manual_seed(seed)
        try:
            mdl_i, preds_val_i = train_one(
                df_tr_f, hp, FEATS_LIST, device
            )
        except Exception as ex:
            log.info(f'Exception while training: {ex}')
            traceback.print_exc()
            return -1

        models.append(mdl_i)

        # convert validation logits → probabilities
        val_probs_i = 1 / (1 + np.exp(-preds_val_i))
        val_stacks.append(val_probs_i)

    

    # ── ensemble distribution on validation ───────────────────────────
    # val_probs_stack = np.vstack(val_stacks)          # (n_models, n_val)
    # ens_val_probs   = val_probs_stack.mean(axis=0)

    # p5, p95 = np.percentile(ens_val_probs, [5, 95])
    # min_th   = max(0.0, p5 - 0.05)
    # max_th   = min(1.0, p95 + 0.05)
    # if max_th - min_th < 0.05:                       # safety fallback
    #     min_th, max_th = 0.20, 0.80

    # steps = max(10, int((int((max_th - min_th) * 100))/3))    # ~1‑pp grid
    # log.info(f"[θ‑cal] search window: {min_th:.3f} – {max_th:.3f}  "
    #         f"({steps} steps)")

    # ── grid‑search θ on train / test / future slices ─────────────────
    # th_tr, score_tr = calibrate_threshold(
    #     df_tr_f, models, hp, FEATS_LIST, device,
    #     min_th=min_th, max_th=max_th, steps=steps
    # )
    # log.info(f"train_th: {th_tr:.4f} → score: {score_tr:.4f}")

    # th_te, score_te = calibrate_threshold(
    #     df_te_f, models, hp, FEATS_LIST, device,
    #     min_th=min_th, max_th=max_th, steps=steps, debug=True
    # )
    # log.info(f"test_th:  {th_te:.4f} → score: {score_te:.4f}")

    # th_fu, score_fu = calibrate_threshold(
    #     df_fu_f, models, hp, FEATS_LIST, device,
    #     min_th=min_th, max_th=max_th, steps=steps
    # )
    # log.info(f"fut_th:   {th_fu:.4f} → score: {score_fu:.4f}")

    # best_th    = float(np.mean([th_te, th_fu]))
    # best_score = float(np.mean([score_te, score_fu]))
    # log.info(f"mean_th:  {best_th:.4f} → mean_score: {best_score:.4f}")

    best_th = 1.2
    hp["threshold"] = 1.2

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

                T = len(df)
                start_idx = hp['seq_len'] - 1
                end_idx   = T - hp['horizon']
                aligned_df = df.iloc[start_idx:end_idx].copy()

                # Backtest on permuted data
                aligned_df_perm, probs, sigs = infer_and_threshold(
                        df_p, best_th, models, hp, FEATS_LIST, device, debug=False
                    )
                trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df, sigs, hp, debug=False)
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
        return 0.5 * v + 0.5 * f

    # ─────────────────────────── 5) BACKTEST TRAIN / TEST SLICES ─────────────
    # ---------- TRAIN ----------
    aligned_df, probs, sigs = infer_and_threshold(
            df_tr_f, best_th, models, hp, FEATS_LIST, device, debug=True
        )
    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df, sigs, hp, debug=True)
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
    aligned_df_te, probs_te, sigs_te = infer_and_threshold(
            df_te_f, best_th, models, hp, FEATS_LIST, device, debug=True
        )
    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df_te, sigs_te, hp, debug=True)
    slices["TEST"] = {
        "stats": stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
        "score": slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
    }

    if slices["TEST"]["score"] <= 0:
        return -1

    # ─────────────────────────── 6) OPTIONAL “FUTURE” SLICE ──────────────────
    aligned_df_fu, probs, sigs_fu = infer_and_threshold(
            df_fu_f, best_th, models, hp, FEATS_LIST, device, debug=True
        )
    trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df_fu, sigs_fu, hp, debug=True)
    log.info("Future Scores")
    slices["FUTURE"] = {
        "stats": stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
        "score": slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars),
    }

    if slices["FUTURE"]["score"] <= 0:
        return -1

    # ─────────────────────────── 7) RANDOM BASELINE (Monte‑Carlo) ────────────
    MC          = 15
    np.random.seed(42)

    n_model_trades = int(sigs_fu.sum())          # match #trades exactly
    rand_scores    = []

    for _ in range(MC):
        idx = np.random.choice(len(aligned_df_fu), size=n_model_trades, replace=False)
        sig_rand      = np.zeros(len(aligned_df_fu), dtype=bool)
        sig_rand[idx] = True

        trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(aligned_df_fu, sig_rand, hp)
        rand_scores.append(
            slice_score(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars, debug=False)
        )

    if not rand_scores:
        mu_rand, sigma_rand = 0.0, 0.05        # benign defaults  (5 % band)
    else:
        mu_rand, sigma_rand = np.mean(rand_scores), np.std(rand_scores, ddof=1)

    sigma_floor = max(0.00001, 0.10 * abs(mu_rand))   # ≥1 % of |μ|
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
    pen_perm, obs_drop = permute_and_score(df_fu_f, FEATS_LIST, models, hp, base_sharpe,
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
            "params":     hp,
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

    syms     = CFG["symbols"]
    interval = "30m"                      # Binance interval string
    start    = CFG["start_date"]         # e.g. '2023-01-01'
    end      = CFG["end_date"]           # e.g. '2024-12-31'

    # ── 2) single-pass optimisation per symbol – no walk-forward windows ─────────
    for sym in syms:
        # pass your CFG dates straight through; fetch_binance_localizes to UTC
        raw = fetch_binance(
            sym,
            interval,
            start,      # '2023-01-01'
            end         # '2024-12-31'
        )

        log.info(raw)

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
                    TimeFrame(30, TimeFrameUnit.Minute),
                    raw_train,
                    raw_test,
                    raw_future,
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