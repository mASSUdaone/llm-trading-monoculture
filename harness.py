#!/usr/bin/env python3
"""
harness.py -- run one generated agent against the frozen tape.

    python3 harness.py agents/agent_001.py
    python3 harness.py agents/agent_001.py --timeout 60 --out runs/agent_001

The agent module must define init(cash) and on_bar(history) at module level and
must not run anything at import time. The harness -- not the agent -- owns the
loop and writes trades.csv, so:

  * the agent physically cannot see a future bar: it is only ever handed
    df.iloc[:b], and rows past b are not in that object;
  * an agent that is killed for exceeding its timeout still leaves behind every
    order it submitted before the kill, so a timed-out agent is distinguishable
    from an agent that simply never traded.

Outputs, per run, into the --out directory:
    trades.csv   timestamp, ticker, direction, size  (timestamp = decision bar)
    run.json     wall-clock, order count, exit status, warnings
"""

import argparse
import importlib.util
import json
import os
import re
import signal
import sys
import time
import traceback

import pandas as pd

CASH = 100_000.0
DEFAULT_TIMEOUT_MIN = 60.0

# Attribute access that could walk from a sliced view back to the parent frame.
# Not a blocker -- a strategy has no reason to touch these -- but flagged so the
# pre-commitment document can say lookahead was verified by inspection.
SUSPICIOUS = [r"\._mgr\b", r"\._data\b", r"\.obj\b", r"\._constructor\b",
              r"\bread_csv\b", r"\bread_parquet\b", r"\bopen\s*\("]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def scan_source(path):
    src = open(path, encoding="utf-8", errors="replace").read()
    return [p.strip("\\b") for p in SUSPICIOUS if re.search(p, src)]


def load_agent(path):
    """Import the module. Anything it runs at import time runs here."""
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("init", "on_bar"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError(f"agent does not define a callable {fn}()")
    return mod


def load_tape():
    if os.path.exists("market.parquet"):
        df = pd.read_parquet("market.parquet")
    else:
        df = pd.read_csv(os.path.join("data", "market.csv"))
    df = df.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
    return df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]


def bar_boundaries(df):
    """Row index one past the end of each distinct timestamp."""
    ts = df["timestamp"].to_numpy()
    ends = [i for i in range(1, len(ts)) if ts[i] != ts[i - 1]] + [len(ts)]
    return ts, ends


def validate(orders, bar_ts, warnings):
    """Keep well-formed orders, record the rest. Never raises."""
    if orders is None:
        return []
    if not isinstance(orders, (list, tuple)):
        warnings.append(f"{bar_ts}: on_bar returned {type(orders).__name__}, not a list")
        return []
    good = []
    for o in orders:
        try:
            ticker, direction, size = o
            direction = str(direction).upper()
            size = int(size)
        except Exception:
            warnings.append(f"{bar_ts}: malformed order {o!r}")
            continue
        if direction not in ("BUY", "SELL") or size <= 0:
            warnings.append(f"{bar_ts}: rejected order {o!r}")
            continue
        good.append((bar_ts, str(ticker), direction, size))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_MIN,
                    help="wall-clock budget in minutes")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    outdir = args.out or os.path.join("runs", os.path.splitext(os.path.basename(args.agent))[0])
    os.makedirs(outdir, exist_ok=True)

    flags = scan_source(args.agent)
    if flags:
        print(f"NOTE: source mentions {flags} -- inspect before trusting this run")

    print("loading tape...", flush=True)
    df = load_tape()
    ts, ends = bar_boundaries(df)
    print(f"{len(df):,} rows | {len(ends):,} bars | {df['ticker'].nunique()} tickers")

    trades, warnings = [], []
    status, detail = "ok", ""
    bars_done = 0

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.timeout * 60))
    t0 = time.time()

    try:
        agent = load_agent(args.agent)
        agent.init(CASH)
        for b in ends:
            bar_ts = ts[b - 1]
            orders = agent.on_bar(df.iloc[:b])
            trades.extend(validate(orders, bar_ts, warnings))
            bars_done += 1
    except Timeout:
        status, detail = "timeout", f"exceeded {args.timeout} min at bar {bars_done}"
    except Exception as e:
        status = "error"
        detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        signal.alarm(0)

    elapsed = time.time() - t0

    pd.DataFrame(trades, columns=["timestamp", "ticker", "direction", "size"]) \
        .to_csv(os.path.join(outdir, "trades.csv"), index=False)

    scored = [t for t in trades if str(t[0]) >= "2025-01-01"]
    meta = {
        "agent": os.path.abspath(args.agent),
        "status": status,
        "detail": detail,
        "wall_clock_sec": round(elapsed, 1),
        "wall_clock_min": round(elapsed / 60, 2),
        "bars_completed": bars_done,
        "bars_total": len(ends),
        "orders_total": len(trades),
        "orders_2025": len(scored),
        "tickers_touched": len({t[1] for t in trades}),
        "source_flags": flags,
        "warnings": warnings[:50],
        "warning_count": len(warnings),
    }
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nstatus            {status} {detail}")
    print(f"wall clock        {elapsed / 60:.2f} min")
    print(f"bars completed    {bars_done:,} / {len(ends):,}")
    print(f"orders submitted  {len(trades):,}  ({len(scored):,} in 2025)")
    print(f"tickers touched   {len({t[1] for t in trades})}")
    if warnings:
        print(f"warnings          {len(warnings)} (see run.json)")
    print(f"\nwrote {outdir}/trades.csv and {outdir}/run.json")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
