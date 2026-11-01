#!/usr/bin/env python3
"""
fetch_tape.py -- build data/market.csv for the monoculture experiment.

Pulls 30-minute SIP bars for a frozen ticker universe over 2024-2025 from
Alpaca. 2024 is burn-in; it stays in the file and is dropped at scoring.

SETUP
    pip3 install alpaca-py pandas pyarrow
    Put your keys in a file called keys.txt next to this script:
        KEY=PKXXXXXXXX
        SECRET=yyyyyyyy
    Put your tickers in universe.txt, one per line.

RUN
    python3 fetch_tape.py

Downloads in small pieces and saves each one into chunks/. If it crashes or
you stop it, just run it again -- it skips whatever it already has.
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

SYMS_PER_REQUEST = 15        # small, so responses stay manageable
MAX_RETRIES = 4
CHUNK_DIR = "chunks"
OUT_CSV = os.path.join("data", "market.csv")
OUT_PARQUET = "market.parquet"
REQUIRE_FULL_COVERAGE = True

# One request per (symbol batch, quarter).
QUARTERS = [
    (datetime(2024, 1, 1), datetime(2024, 4, 1)),
    (datetime(2024, 4, 1), datetime(2024, 7, 1)),
    (datetime(2024, 7, 1), datetime(2024, 10, 1)),
    (datetime(2024, 10, 1), datetime(2025, 1, 1)),
    (datetime(2025, 1, 1), datetime(2025, 4, 1)),
    (datetime(2025, 4, 1), datetime(2025, 7, 1)),
    (datetime(2025, 7, 1), datetime(2025, 10, 1)),
    (datetime(2025, 10, 1), datetime(2026, 1, 1)),
]


def load_keys(path="keys.txt"):
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    if key and secret:
        return key, secret
    if not os.path.exists(path):
        print(f"error: no {path} found, and ALPACA_KEY/ALPACA_SECRET are not set.\n"
              f"Create {path} next to this script containing two lines:\n"
              f"  KEY=PKxxxxxxxx\n  SECRET=yyyyyyyy", file=sys.stderr)
        sys.exit(1)
    vals = {}
    for ln in open(path):
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            vals[k.strip().upper()] = v.strip()
    if "KEY" not in vals or "SECRET" not in vals:
        print(f"error: {path} needs a KEY= line and a SECRET= line", file=sys.stderr)
        sys.exit(1)
    return vals["KEY"], vals["SECRET"]


def load_universe(path="universe.txt"):
    if not os.path.exists(path):
        print(f"error: {path} not found (one ticker per line)", file=sys.stderr)
        sys.exit(1)
    syms = [ln.strip().upper() for ln in open(path) if ln.strip()]
    return sorted(set(s.replace("-", ".") for s in syms))


def fetch_one(client, batch, start, end):
    """One request, retried with backoff. Returns a DataFrame (may be empty)."""
    req = StockBarsRequest(
        symbol_or_symbols=batch,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.SIP,          # free for data older than 15 minutes
        adjustment=Adjustment.ALL,  # correct for splits and dividends
    )
    for attempt in range(MAX_RETRIES):
        try:
            return client.get_stock_bars(req).df
        except Exception as e:
            msg = str(e)[:90]
            if attempt == MAX_RETRIES - 1:
                print(f"      giving up: {msg}")
                return pd.DataFrame()
            wait = 3 * (attempt + 1)
            print(f"      retry {attempt + 1} in {wait}s ({msg})")
            time.sleep(wait)
    return pd.DataFrame()


EMPTY_COLS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def download(client, symbols):
    os.makedirs(CHUNK_DIR, exist_ok=True)
    batches = [symbols[i:i + SYMS_PER_REQUEST]
               for i in range(0, len(symbols), SYMS_PER_REQUEST)]
    total = len(batches) * len(QUARTERS)
    done = 0

    for bi, batch in enumerate(batches):
        for qi, (start, end) in enumerate(QUARTERS):
            done += 1
            path = os.path.join(CHUNK_DIR, f"b{bi:03d}_q{qi}.parquet")
            if os.path.exists(path):
                continue
            print(f"  [{done}/{total}] batch {bi + 1} {start:%Y-%m}", flush=True)
            df = fetch_one(client, batch, start, end)
            if len(df):
                df.reset_index().to_parquet(path, index=False)
            else:
                # empty marker, so a rerun doesn't retry this forever
                pd.DataFrame(columns=EMPTY_COLS).to_parquet(path, index=False)
            time.sleep(0.3)


def assemble():
    frames = []
    for f in sorted(os.listdir(CHUNK_DIR)):
        d = pd.read_parquet(os.path.join(CHUNK_DIR, f))
        if len(d):
            frames.append(d)
    if not frames:
        print("no data downloaded", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"symbol": "ticker"})
    return df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]


def clean(df):
    # Alpaca returns UTC and includes extended-hours bars. Keep only the
    # 09:30-15:30 ET bar starts -- 13 bars per regular session.
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.assign(timestamp=ts)
    mins = ts.dt.hour * 60 + ts.dt.minute
    df = df[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 30) & (ts.dt.dayofweek < 5)]
    df = df.drop_duplicates(subset=["timestamp", "ticker"])

    # Keep only full 13-bar sessions, so every ticker sees the same grid.
    sess = df["timestamp"].dt.date
    per = df.assign(s=sess).groupby("s")["timestamp"].nunique()
    full = set(per[per == 13].index)
    print(f"sessions kept: {len(full)} (dropped {len(per) - len(full)} partial/holiday)")
    df = df[sess.isin(full)]

    if REQUIRE_FULL_COVERAGE:
        nbars = df["timestamp"].nunique()
        counts = df.groupby("ticker")["timestamp"].nunique()
        keep = counts[counts == nbars].index
        print(f"full coverage: {len(keep)} / {df['ticker'].nunique()} tickers")
        df = df[df["ticker"].isin(keep)]

    df = df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)


def main():
    key, secret = load_keys()
    symbols = load_universe()
    print(f"universe: {len(symbols)} tickers")

    client = StockHistoricalDataClient(key, secret)
    download(client, symbols)

    print("\nassembling...")
    df = clean(assemble())

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{len(df):,} rows | {df['timestamp'].nunique():,} bars | "
          f"{df['ticker'].nunique()} tickers")
    print(f"wrote {OUT_CSV} and {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
