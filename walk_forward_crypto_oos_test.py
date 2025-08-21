import json
import logging
import warnings
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tabulate import tabulate

from walk_forward_crypto import PriceDirectionModel, FEATS_LIST, fetch_binance, build_feats, infer_and_threshold, run_backtest, stats

# silence unavoidable NumPy “divide by zero” notes
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────── config ──────────────────────────────────
CFG         = yaml.safe_load(Path("config.yaml").read_text())
API_KEY     = CFG["api_key"];  SECRET_KEY = CFG["secret_key"]

TEST_START = "2025-04-01"
TEST_END   = "2025-05-26"

MODEL_DIR = Path("trained_models")
PARAM_DIR = Path("best_params")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

FEAT_CACHE: dict[str, dict[int, pd.DataFrame]] = defaultdict(dict)

def main() -> None:
    """
    O-O-S evaluation on TEST_START→TEST_END using 3-model ensemble.
    Now also tracks and prints the last 20 (most‑recent) predictions/signals
    per symbol in the requested CSV‑style format **without affecting any
    existing evaluation logic**.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device → {device}")

    live_rows = []

    # ────────────────────────────────────────────────────────────────────────────────
    # NEW: container for the last‑20 signal snapshots per symbol
    # ────────────────────────────────────────────────────────────────────────────────
    from collections import defaultdict, deque
    last_signals = defaultdict(lambda: deque(maxlen=20))  # symbol → recent rows

    # iterate over ensemble pickle files
    for ens_file in sorted(MODEL_DIR.glob("best_*_ensemble.pkl")):
        tag = ens_file.stem.replace("best_", "").replace("_ensemble", "")
        sym, tfstr = tag.split("_", 1)
        sym = sym[0:3] + "/" + sym[3:]
        interval = "30m"

        # load hyperparams from JSON
        hp_path = PARAM_DIR / f"best_params_{tag}.json"
        details = json.loads(hp_path.read_text())
        hp = details["Aggregate"]["HyperParams"]

        # reconstruct ensemble models and isotonic calibrators
        ensemble = pickle.load(open(ens_file, "rb"))  # list of checkpoints
        models = []
        for chk in ensemble:
            model = PriceDirectionModel(len(FEATS_LIST), hp["hidden_size"],
                                hp["num_layers"], hp["dropout"]).to(device)
            model.load_state_dict(chk["state_dict"])
            model.eval()
            models.append(model)

        # fetch raw data with pre-buffer for features
        raw_full = fetch_binance(sym, interval, TEST_START, TEST_END)
        raw_full = raw_full[~raw_full.index.duplicated(keep="last")]
        if raw_full.empty:
            log.warning(f"No data for {sym}; skipping.")
            continue

        # feature build & cache
        cache = build_feats(raw_full, hp["ma_window"], sym)
        FEAT_CACHE[sym][hp["ma_window"]] = cache

        # slice TEST window
        raw_test = raw_full.loc[TEST_START:TEST_END]
        #log.info(f'last 20 of raw_test: {raw_test.tail(20)}')
        raw_test.dropna(how="any").loc[lambda d: ~d.index.duplicated(keep="last")]
        feat_tmp = cache.reindex(raw_test.index)
        feat_full = feat_tmp.dropna(how="any").loc[lambda d: ~d.index.duplicated(keep="last")]

        #log.info(f'last 20 of feat_full: {feat_full.tail(20)}')

        # guard on seq_len + horizon
        required = hp["seq_len"] + hp["horizon"]
        if len(feat_full) < required:
            log.info(f"Not enough bars for {sym}: have {len(feat_full)}, need {required}; skipping.")
            continue

        # infer & threshold using ensemble
        threshold = hp.get("threshold", 0.0)
        aligned_df, probs, sigs = infer_and_threshold(
            feat_full, threshold, models, hp, FEATS_LIST, device, debug=True
        )

        #log.info(f'last 20 of sub: {sub.tail(20)}')

        n_model_trades = int(sigs.sum())          # match #trades exactly

        idx = np.random.choice(len(aligned_df), size=n_model_trades, replace=False)
        sig_rand      = np.zeros(len(aligned_df), dtype=bool)
        sig_rand[idx] = True

        # backtest original signals + variants
        sig_sets = {
            "orig": sigs,
            "inv": ~sigs,
            "R50": sig_rand
        }

        for name, sig in sig_sets.items():

            log.info(f"Running backtest for {name}")

            log.info(f'Sigs: {sig}')

            trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars = run_backtest(
                aligned_df, sig, hp, debug=True
            )

            metrics = stats(trades, trade_rate, win_rate, avg_R_winners, avg_R_losers, expectancy_R, ending_balance, pnl, percent_gains, sharpe, avg_hold_bars)

            live_rows.append({
                "Model":        f"{tag}_{name}",
                "Symbol":       sym,
                "Env":          "LIVE",
                "Trades":       metrics["Trades"],
                "TradeRate":    metrics["TradeRate"],
                "WinRate":      metrics["WinRate"],
                "avg_R_winners":   metrics["avg_R_winners"],
                "avg_R_losers": metrics["avg_R_losers"],
                "expectancy_R":       metrics["expectancy_R"],      # per‑trade Sharpe (R‑based)
                "percent_gains":  metrics["percent_gains"],     # in PnL units
                "sharpe":  metrics["sharpe"],     # in PnL units
                "avg_hold_bars":  metrics["avg_hold_bars"],
            })

    # summary & output ──────────────────────────────────────────────────────────────
    if live_rows:
        df = pd.DataFrame(live_rows).sort_values(["expectancy_R", "sharpe"], ascending=False)
        print("\n=== O-O-S LIVE ===")
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))
        df.to_csv("oos_live.csv", index=False)
        log.info("Saved: oos_live.csv")

    # NEW: print the compiled last‑20‑signals table (per symbol) ────────────────
    # if last_signals:
    #     sig_records = [rec for dq in last_signals.values() for rec in dq]
    #     df_sig = (
    #         pd.DataFrame(sig_records)
    #           .sort_values(["symbol", "bar_close"], ascending=[True, True])
    #           .reset_index(drop=True)
    #     )
    #     print("\n=== LAST 20 SIGNALS (PER SYMBOL) ===")
    #     print(tabulate(df_sig, headers="keys", tablefmt="github", showindex=False, floatfmt=".6f"))
    #     # Persist for post‑run inspection (optional)
    #     df_sig.to_csv("last_signals.csv", index=False)
    #     log.info("Saved: last_signals.csv")

if __name__ == "__main__":
    main()