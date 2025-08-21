#!/usr/bin/env python3
"""
LIVE PAPER TRADER – 30-minute swing-scalper (refactored)
===============================================================
Streams 1-min bars → rolls to 30-min
Uses each best_*.pth model (ATR-threshold signal)
Risk per trade, per-symbol capital buckets
Bracket orders (TP/SL) simulated in paper/live
Horizon exits after `horizon` bars
Detailed logging: aggregation, orders, P&L, win-rate
Crash recovery via state persistence
"""
import asyncio
import json
import logging
import signal
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
import pandas_market_calendars as mcal
import pytz
import pickle
import csv
from typing import Dict, List, Tuple, Set, Optional, Any, DefaultDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest, ClosePositionRequest, GetOrdersRequest, QueryOrderStatus
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.stream import TradingStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderType
from alpaca.data.live.crypto import CryptoDataStream, CryptoFeed
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.common.exceptions import APIError

from math import sqrt

from walk_forward_crypto import FEATS_LIST, build_feats, fetch, infer_and_threshold, LSTMRegSimple, ROLL_SPAN_DEFAULT

# ───── USER CONFIG ──────────────────────────────────────────────────────────
API_KEY           = "PKW0DSC5V72B4HCLP6FV"
SECRET_KEY        = "1sZy3MZVk15uIILxepF9JBQeN3hNXehnEyGRlPKc"
PAPER             = True               # use paper trading keys

COST_FRAC         = 0.20               # fraction of *remaining* bucket to cap cost per trade
RISK_FRAC         = 0.01               # fraction of *remaining* bucket to cap cost per trade

MODEL_DIR         = Path("trained_models")
PARAM_DIR         = Path("best_params")

# state file for crash recovery
STATE_FILE        = Path("state.json")

# ───── LOGGING ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s"
)
log = logging.getLogger("live")

# ───── PERSISTENCE ─────────────────────────────────────────────────────────
def save_state():
    serial_open = {}
    for sym, poses in OPEN.items():
        serial_open[sym] = []
        for p in poses:
            p2 = p.copy()
            p2["expiry"] = p2["expiry"].isoformat()
            serial_open[sym].append(p2)
    state = {
        "open":  serial_open,
        "alloc": ALLOC_PER_SYMBOL,
        "stats": {k: v for k, v in STATS.items()}   # cast off defaultdict
    }
    STATE_FILE.write_text(json.dumps(state, indent=4, default=str))  # Added indent=4
    log.info("State saved")


def load_state():
    if not STATE_FILE.exists():
        return
    data = json.loads(STATE_FILE.read_text())
    for sym, poses in data.get("open", {}).items():
        OPEN[sym] = []
        for p in poses:
            p2 = p.copy()
            p2["expiry"] = datetime.fromisoformat(p2["expiry"])
            OPEN[sym].append(p2)
    ALLOC_PER_SYMBOL.update(data.get("alloc", {}))
    STATS.update(data.get("stats", {}))
    log.info("State loaded")

# ───── LOAD MODELS ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models[tag]  →  (list[nn.Module], list[IsotonicRegression|None], hp_dict)
models: Dict[str, Tuple[List[nn.Module], List[object], dict]] = {}

for pk in MODEL_DIR.glob("best_*_ensemble.pkl"):
    log.info(f"pk: {pk}")
    tag = pk.stem.removeprefix("best_").removesuffix("_ensemble")

    # hyper‑params
    p_js = PARAM_DIR / f"best_params_{tag}.json"
    if not p_js.exists():
        log.warning(f"No hyper‑param file for {tag}; skipping.")
        continue
    hp = json.loads(p_js.read_text())["Aggregate"]["HyperParams"]

    # reconstruct ensemble
    ensemble_ckpts = pickle.load(open(pk, "rb"))     # list[dict]
    mdl_list, iso_list = [], []
    for chk in ensemble_ckpts:
        mdl = LSTMRegSimple(
            len(FEATS_LIST),           # input channels
            hp["hidden_dim"],
            hp["num_layers"],
            hp["dropout"]
        ).to(device)
        mdl.load_state_dict(chk["state_dict"])
        mdl = mdl.float().eval()
        mdl_list.append(mdl)
        iso_list.append(chk.get("isotonic", None))

    models[tag] = (mdl_list, iso_list, hp)
    log.info(f"Loaded ensemble {tag}  (#models={len(mdl_list)})")

if not models:
    log.error("No ensembles found. Exiting.")
    sys.exit(1)

symbols = [
    f"{t.split('_', 1)[0][:3]}/{t.split('_', 1)[0][3:]}"
    for t in models
]

LIVE_STORE = {}

for s in symbols:
    LIVE_STORE[s] = Path(f"live_preds_{s.replace('/', '')}.csv")
    if not LIVE_STORE[s].exists():
        with LIVE_STORE[s].open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol", "bar_close", "open", "high", "low", "close", "volume",
                "pred", "signal", "threshold", "recorded_at"
            ])

# ───── CAPITAL CONTROL ─────────────────────────────────────────────────────
OPEN: Dict[str, List[dict]] = defaultdict(list)

# ───── STATS & HISTORY ─────────────────────────────────────────────────────
STATS: DefaultDict[str, Dict[str, Any]] = defaultdict(
    lambda: {"trades": 0, "wins": 0, "returns": []}   # <-- include “returns”
)
ROLL30: Dict[str, pd.DataFrame]  = {s: pd.DataFrame() for s in symbols}

MAX_30M_BARS     = 10_000         # how many historical 30-min bars to seed with

# ───── CLIENT SETUP ────────────────────────────────────────────────────────
trader   = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
tstream  = None
acct     = trader.get_account()
log.info(f"Paper equity ${acct.equity}")

acct        = trader.get_account()
TOTAL_EQ    = float(acct.equity) * 0.5
N_SYMBOLS   = len(symbols)
log.info(f"Paper equity ${TOTAL_EQ:.2f} split across {N_SYMBOLS} symbols")

# initialize per-symbol buckets as 1/N of total equity
ALLOC_PER_SYMBOL = { s: TOTAL_EQ / N_SYMBOLS for s in symbols }
log.info(f"Initial bucket per symbol: ${TOTAL_EQ/N_SYMBOLS:.2f}")

def record_trade(sym: str, entry_px: float, exit_px: float, type: str) -> None:
    """Update STATS after a position is fully closed."""
    r = (exit_px - entry_px) / entry_px          # raw (not %) P&L for a long
    log.info(f"{sym}: {type} trade completed, returns: {r}")
    st = STATS[sym]
    st["trades"] += 1
    if r > 0:
        st["wins"] += 1
    st["returns"].append(r)
    # optional: cap history so the file never grows without bound
    if len(st["returns"]) > 5_000:
        st["returns"] = st["returns"][-5_000:]

# ───── HISTORICAL FETCH – 30-MIN BARS ──────────────────────────────────────
def preload_30m_history(sym: str, n: int = MAX_30M_BARS) -> pd.DataFrame:
    now      = datetime.now(timezone.utc)
    boundary = now.replace(second=0,microsecond=0) - timedelta(minutes=now.minute % 30)
    start    = boundary - timedelta(minutes=30 * n)
    log.info(f"Start: {start}")
    log.info(f"Boundary: {boundary}")
    # now just delegate to fetch()
    df = fetch(sym,
               TimeFrame(30, TimeFrameUnit.Minute),
               start, boundary)
    return df.tail(n)

def compute_bar_expiry(bar_ts: pd.Timestamp, horizon_bars: int) -> pd.Timestamp:
    """
    Starting at bar_ts (UTC), count forward `horizon_bars` 30-min
    intervals that fall within NYSE open hours, and return that timestamp.
    """

    return bar_ts + timedelta(minutes=30 * horizon_bars)

# ─── constants ──────────────────────────────────────────────────────────
MAX_ATTEMPTS    = 30          # how many times to poll for the finished bar
RETRY_INTERVAL  = 30          # seconds to wait between attempts

# ─── helpers ────────────────────────────────────────────────────────────
async def fetch_and_process(sym: str, boundary: datetime):
    """
    Retrieve the completed 30‑minute bar that closed at *boundary* and hand it
    to process_30m().  Retries until it appears or MAX_ATTEMPTS is reached.

    Parameters
    ----------
    sym       : the instrument symbol
    boundary  : a datetime aligned on a 30‑minute boundary, e.g. 17:30:00 UTC
    """
    expected_ts = boundary - timedelta(minutes=30)          # 17:00 bar

    for attempt in range(1, MAX_ATTEMPTS + 1):
        ts_now = datetime.now(timezone.utc)

        # Run the (blocking) fetch in a thread pool
        df = await asyncio.get_event_loop().run_in_executor(
            None,
            fetch,
            sym,
            TimeFrame(30, TimeFrameUnit.Minute),
            expected_ts,            # start at the bar we actually need
            ts_now                  # up to "now"
        )

        if expected_ts in df.index:                     # bar is ready
            bar30 = df.loc[expected_ts]
            log.info(
                f"{sym}: delivered 30‑min bar @{expected_ts} "
                f"(attempt {attempt})"
            )
            process_30m(sym, bar30)
            return

        log.warning(
            f"{sym}: {expected_ts} bar not available yet "
            f"(attempt {attempt}/{MAX_ATTEMPTS})"
        )
        await asyncio.sleep(RETRY_INTERVAL)

    log.error(
        f"{sym}: giving up after {MAX_ATTEMPTS} attempts – "
        f"bar {expected_ts} never arrived"
    )


async def clock_30m():
    """
    Drive the 30‑minute workflow: wait until one minute *after* each
    :00 / :30 boundary, then launch a fetch_and_process() task for every
    symbol.
    """
    while True:
        now = datetime.now(timezone.utc)

        # Next boundary (either :30 or top‑of‑hour)
        if now.minute < 30:
            boundary = now.replace(minute=30, second=0, microsecond=0)
        else:
            boundary = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )

        # Sleep until one minute after that boundary so the provider
        # has time to close & publish the bar
        to_wait = (boundary - now).total_seconds() + 60
        await asyncio.sleep(to_wait)

        # Kick off one task per symbol for the bar that just closed
        tasks = [
            asyncio.create_task(fetch_and_process(sym, boundary))
            for sym in symbols
        ]
        await asyncio.gather(*tasks)

def wait_for_fill(order: AlpacaOrder, qty: float, timeout: float = 30.0, poll_interval: float = 0.5) -> AlpacaOrder:
    """
    Polls Alpaca for up to `timeout` seconds until `order.filled_qty` >= `qty`.
    Returns the latest Order object (filled or not).
    """
    start = time.time()
    while True:
        # assuming trader.get_order exists; if not, switch to get_orders filtering by ID
        try:
            current = trader.get_order_by_id(order.id)
        except AttributeError:
            # fallback: list all orders and pick ours
            orders = trader.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL))
            current = next((o for o in orders if o.id == order.id), order)
        filled = float(current.filled_qty or 0)
        if filled >= qty:
            return current
        if time.time() - start > timeout:
            return current
        time.sleep(poll_interval)

def record_live_pred(
    sym: str,
    bar30: pd.Series,
    pred: float,
    sig: bool,
    threshold: float
):
    """Append one row of live-prediction info to CSV, including the symbol."""
    with LIVE_STORE[sym].open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            sym,
            bar30.name.isoformat(),
            float(bar30.open),
            float(bar30.high),
            float(bar30.low),
            float(bar30.close),
            float(bar30.volume),
            pred,
            int(sig),
            threshold,
            datetime.utcnow().isoformat(),
        ])

def process_30m(sym: str, bar30: pd.Series) -> None:
    """
    Live 30-minute handler for crypto.
     0) Horizon-based forced exits at market
     1) … rest of your existing logic …
    """

    # ───────────────────────────── 1) locate model ─────────────────────────────
    tag, (mdl_list, iso_list, hp) = next(
        (t, models[t]) for t in models
        if t.startswith(sym.replace("/", "") + "_")
    )
    #log.info(f"{sym}: ▶ process_30m start  tag={tag}")

    # ───────────────────────────── 2) log the bar ─────────────────────────────
    ts = bar30.name
    log.info(
        f"{sym}: bar @ {ts.isoformat()}  O={bar30.open:.2f}  H={bar30.high:.2f}  "
        f"L={bar30.low:.2f}  C={bar30.close:.2f}  V={bar30.volume}"
    )

    # ─────────────────────────── 3) rolling buffer ─────────────────────────────
    ROLL30[sym] = ROLL30[sym]._append(bar30).tail(2_000)

    # ───────────────────────── 4) feature construction ─────────────────────────
    feats_df = build_feats(ROLL30[sym], hp["ma_window"], sym)
    feats_df = (
        feats_df
        .dropna()                                                # no missing feature rows
        .loc[~feats_df.index.duplicated(keep="last")]           # drop duplicate timestamps
    )
    need = ROLL_SPAN_DEFAULT + hp["seq_len"] + hp["horizon"] + 100
    if len(feats_df) < need:
        log.info(f"{sym}: insufficient data ({len(feats_df)}/{need})")
        return

    # ─────────────────────────── 5) inference & filter ─────────────────────────
    tf_str  = tag.split("_", 1)[1]
    device0 = next(mdl_list[0].parameters()).device
    _, preds, sigs, tp_arr, sl_arr = infer_and_threshold(
        feats_df, hp.get("threshold", 0.0),
        mdl_list, hp, sym, tf_str,
        FEATS_LIST, device0, iso_list, debug=False,
    )

    # Grab only the latest bar’s prediction
    latest_pred  = float(preds[-1])
    latest_sig   = bool(sigs[-1])
    latest_thres = float(hp.get("threshold", 0.0))

    # Always record—even if no signal—so you can later compare
    record_live_pred(sym, bar30, latest_pred, latest_sig, latest_thres)

    if not sigs[-1]:
        log.info(f"{sym}: no signal on latest bar, sig: {sigs[-1]} pred: {preds[-1]}")
        return

    pred   = float(preds[-1])
    sl_pct = float(sl_arr[-1])
    tp_pct = float(tp_arr[-1])

    log.info(f"{sym}: signal → pred={pred:.4f}, sl_pct={sl_pct:.2f}%, tp_pct={tp_pct:.2f}%")    

    # ─────────────────────────── 6) get SL/TP ―────────────────────────────────
    sl_frac = float(sl_pct) / 100.0
    tp_frac = float(tp_pct) / 100.0
    if sl_frac <= 0:
        log.warning(f"{sym}: SL ≤ 0, abort")
        return

    price  = bar30.close
    bucket = ALLOC_PER_SYMBOL[sym]

    # ────────────────────────── 7) budgets & balances ─────────────────────────
    risk_b    = bucket * RISK_FRAC
    cost_b    = bucket * COST_FRAC
    #log.info(f"{sym}: risk_budget={risk_b:.2f}, cost_budget={cost_b:.2f}")

    risk_unit = price * sl_frac
    #log.info(f"{sym}: risk_per_unit={risk_unit:.2f}")

    # ───────────────────── exchange increments ─────────────────────────────────
    asset = trader.get_asset(sym)
    step  = (
        hp.get("qty_step") or
        getattr(asset, "min_trade_increment", None) or
        getattr(asset, "min_order_size", None) or
        1e-6
    )
    if step <= 0: step = 1e-6

    tick = (
        hp.get("price_tick") or
        getattr(asset, "price_increment", None) or
        0.01
    )
    dec = max(0, int(-math.log10(tick)))
    #log.info(f"{sym}: qty_step={step}, price_tick={tick}, dec={dec}")

    # ─────────────────────────── 8) compute qty ────────────────────────────────
    max_r = risk_b   / risk_unit
    max_c = cost_b   / price
    raw_q = min(max_r, max_c)
    qty   = math.floor(raw_q / step) * step
    if qty < step:
        log.info(f"{sym}: qty {qty:.6f} < step {step:.6f}, skip")
        return

    # ────────────────────────── 9) TP/SL prices ───────────────────────────────
    tp_px = round(price * (1 + tp_frac), dec)
    sl_px = round(price * (1 - sl_frac), dec)
    log.info(f"{sym}: BUY {qty:.6f} @ {price:.2f}, TP {tp_px}, SL {sl_px}")

    # 0) snapshot pre-buy via get_open_position (raises APIError if no pos)
    try:
        pre_pos = trader.get_open_position(sym.replace('/', ''))
        pre_qty = float(pre_pos.qty)
    except APIError:
        pre_qty = 0.0

    # 1) submit & wait synchronously for fill
    entry = trader.submit_order(
        MarketOrderRequest(
            symbol        = sym,
            side          = OrderSide.BUY,
            type          = OrderType.MARKET,
            qty           = qty,
            time_in_force = TimeInForce.GTC,
        )
    )
    filled_order = wait_for_fill(entry, qty)  # blocking, no await
    bought       = float(filled_order.filled_qty or 0)
    avg_px       = float(filled_order.filled_avg_price or 0)

    if bought <= 0:
        log.error(f"{sym}: no fill after buy order {entry.id}")
        return
    
    log.info(f'Expected to buy at px: {price}, market price was: {avg_px}')

    # 2) snapshot post-buy position
    try:
        post_pos = trader.get_open_position(sym.replace('/', ''))
        post_qty = float(post_pos.qty)
    except APIError:
        log.error(f"{sym}: expected position after buy, got none")
        return

    # 3) compute net received
    net_bought = post_qty - pre_qty
    if net_bought <= 0:
        log.error(f"{sym}: net change ≤ 0 (pre={pre_qty}, post={post_qty})")
        return
    if net_bought < bought:
        log.warning(
            f"{sym}: requested {qty:.6f}, raw fill {bought:.6f}, "
            f"net position change {net_bought:.6f}"
        )

    # 4) record using net_bought
    ALLOC_PER_SYMBOL[sym] -= net_bought * avg_px
    OPEN[sym].append({
        "entry_order_id": entry.id,
        "entry_px":       avg_px,
        "qty":            net_bought,
        "qty_remaining":  net_bought,
        "tp_px":          tp_px,
        "sl_px":          sl_px,
        "expiry":         compute_bar_expiry(bar30.name, hp["horizon"]),
    })
    save_state()
    summarize_stats()
    log.info(f"{sym}: BUY recorded {net_bought:.6f} @ {avg_px:.2f}")

async def process_1m(sym: str, bar1m: pd.Series) -> None:
    """
    1-minute handler: checks TP/SL and horizon exit on each new bar.
    Exits via market orders for immediate execution, then uses actual fill price.
    """
    now = bar1m.name
    high, low = bar1m.high, bar1m.low

    for pos in list(OPEN[sym]):
        qty = pos["qty_remaining"]
        tp_px = pos["tp_px"]
        sl_px = pos["sl_px"]

        # common function to submit, wait, record, and clean up
        async def exit_position(reason: str):
            log.info(f"{sym}: {reason} triggered, market-selling {qty:.6f} BTC")
            entry = trader.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    qty=qty,
                    time_in_force=TimeInForce.GTC,
                )
            )
            filled_order = wait_for_fill(entry, qty)
            filled = float(filled_order.filled_qty or 0)
            avg_px = float(filled_order.filled_avg_price or 0)

            if filled <= 0 or avg_px <= 0:
                log.error(f"{sym}: no fill after timeout for sell order {entry.id}, aborting TP/SL")
                return
            
            log.info(f'{sym}: Market Sell Filled for: {avg_px}')

            # update allocation & records with real exit price
            ALLOC_PER_SYMBOL[sym] += filled * avg_px
            record_trade(sym, pos["entry_px"], avg_px, reason)
            OPEN[sym].remove(pos)
            save_state()
            summarize_stats()

        # take-profit
        if high >= tp_px:
            log.info(f'{sym}: Taking profit hoping for px: {high}')
            await exit_position("take profit")
            continue

        # stop-loss
        if low <= sl_px:
            log.info(f'{sym}: Stop loss hoping for px: {low}')
            await exit_position("stop loss")
            continue

        # horizon expiry
        if now >= pos["expiry"]:
            log.info(f'{sym}: Horizon Exit hoping for good market price')
            await exit_position("horizon exit")
            continue

# ───── UPDATED SUMMARIZE STATS ─────────────────────────────────────────────
def summarize_stats(window: int = 50):
    """
    For each symbol, print:
      • total trades, win rate, expectancy
      • rolling Sharpe, max drawdown, profit factor over last `window` trades
    """
    for s, st in STATS.items():
        trades = st.get("trades", 0)
        wins   = st.get("wins", 0)
        returns = st.get("returns", [])

        # basic stats
        win_rate   = (wins / trades * 100) if trades else 0.0
        expectancy = (np.mean(returns) * 100) if returns else 0.0

        # rolling window
        recent = returns[-window:] if len(returns) >= window else returns
        n = len(recent)
        if n >= 2:
            mean_r = np.mean(recent)
            std_r  = np.std(recent, ddof=1)
            # non-annualized Sharpe scaled by √n
            sharpe = (mean_r / std_r) * sqrt(n) if std_r > 0 else 0.0

            # equity curve & max drawdown
            eq_curve = np.cumsum(recent)
            peaks    = np.maximum.accumulate(eq_curve)
            drawdowns = peaks - eq_curve
            max_dd   = np.max(drawdowns)

            # profit factor
            wins_sum  = sum(r for r in recent if r > 0)
            loss_sum  = -sum(r for r in recent if r < 0) or 1e-8
            pf        = wins_sum / loss_sum
        else:
            sharpe = max_dd = pf = 0.0

        log.info(
            f"{s}  "
            f"trades={trades}, win%={win_rate:.1f}%, exp={expectancy:.2f}% | "
            f"Sharpe≈{sharpe:.2f}, MaxDD={max_dd:.2%}, PF={pf:.2f}"
        )

async def _handle_1m_bar(bar):
    # convert the incoming Bar into the pd.Series your process_1m expects
    bar1m = pd.Series(
        {
            "open":   bar.open,
            "high":   bar.high,
            "low":    bar.low,
            "close":  bar.close,
            "volume": bar.volume,
        },
        name=bar.timestamp,
    )
    await process_1m(bar.symbol, bar1m)

async def runner():
    """
    Start the crypto data stream and 30m clock.
    """
    # subscribe to 1m bars for exit logic
    tstream.subscribe_bars(_handle_1m_bar, *symbols)

    # run both the 30m clock and the websocket
    await asyncio.gather(
        clock_30m(),
        tstream._run_forever()
    )

def main():
    """
    Initialize state, seed history, and start the runner.
    """
    # 1) load state
    load_state()

    # 2) seed 30m history for each symbol
    for s in symbols:
        pre = preload_30m_history(s)
        if not pre.empty:
            ROLL30[s] = pre.copy()
            log.info(f"{s}: seeded {len(pre)} 30-min bars (last={pre.index[-1]})")
        else:
            log.warning(f"{s}: no 30-min history loaded")

    # 3) init crypto data stream
    global tstream
    tstream = CryptoDataStream(API_KEY, SECRET_KEY, feed=CryptoFeed.US)

    # 4) start async runner
    asyncio.run(runner())

# ───── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────
def shutdown(*_):
    log.info("Shutting down…")
    save_state()
    sys.exit(0)

if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)
    main()
