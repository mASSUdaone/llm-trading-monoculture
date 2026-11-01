#!/usr/bin/env python3
"""
rebuild.py -- rebuild data/market.csv from the already-downloaded chunks/
folder, using a coverage threshold instead of requiring a price in every bar.

Run with no arguments to see how many tickers survive at various thresholds:
    python3 rebuild.py

Run with a threshold to actually write the file:
    python3 rebuild.py 0.98
"""

import os
import sys

import pandas as pd

CHUNK_DIR = "chunks"
OUT_CSV = os.path.join("data", "market.csv")
OUT_PARQUET = "market.parquet"


def assemble():
    frames = []
    for f in sorted(os.listdir(CHUNK_DIR)):
        d = pd.read_parquet(os.path.join(CHUNK_DIR, f))
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"symbol": "ticker"})
    return df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]


def prepare(df):
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.assign(timestamp=ts)
    mins = ts.dt.hour * 60 + ts.dt.minute
    df = df[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 30) & (ts.dt.dayofweek < 5)]
    df = df.drop_duplicates(subset=["timestamp", "ticker"])

    sess = df["timestamp"].dt.date
    per = df.assign(s=sess).groupby("s")["timestamp"].nunique()
    full = set(per[per == 13].index)
    df = df[sess.isin(full)]
    return df


def main():
    print("reading chunks...")
    df = prepare(assemble())
    nbars = df["timestamp"].nunique()
    counts = df.groupby("ticker")["timestamp"].nunique()
    frac = counts / nbars

    print(f"\n{nbars:,} bars total, {len(counts)} tickers downloaded\n")
    print("  threshold   tickers kept")
    for t in [1.00, 0.999, 0.99, 0.98, 0.95, 0.90, 0.80]:
        print(f"    {t:>5.3f}       {(frac >= t).sum():>4}")

    if len(sys.argv) < 2:
        print("\nPick one, then rerun:  python3 rebuild.py 0.98")
        return 0

    thresh = float(sys.argv[1])
    keep = frac[frac >= thresh].index
    df = df[df["ticker"].isin(keep)]
    df = df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    df = df.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    missing = 1.0 - len(df) / (df["ticker"].nunique() * nbars)
    print(f"\n{len(df):,} rows | {df['timestamp'].nunique():,} bars | "
          f"{df['ticker'].nunique()} tickers")
    print(f"{missing:.2%} of ticker-bar slots are empty (a ticker with no trade "
          f"in that half hour simply has no row)")
    print(f"wrote {OUT_CSV} and {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
