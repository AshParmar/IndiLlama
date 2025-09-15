#!/usr/bin/env python
"""
Balance a combined Marathi sentiment dataset containing multiple domains (e.g., Movie Reviews 'MR' and Subtitles/Tweets 'MS').

Input CSV is expected to have at least columns:
  text, label, source
Optionally: label_raw

Balancing Modes:
  strict-domain    : Enforce equal counts for each (label, source) pair. For every label L, find the minimum count across sources, and sample that many rows per source.
  label-only       : Balance only by label globally (ignore domain). Takes min class count and samples that many from each label.
  proportional     : No balancing; just copies the original (after optional shuffling) for reference.

Splitting:
  After balancing (except proportional), create train/val/test splits with ratios 70/15/15 using stratification:
    - strict-domain: stratify on combined key label|source
    - label-only   : stratify on label

Output Files (written to same directory as input by default):
  balanced_mode_<mode>.csv
  train_<mode>.csv
  val_<mode>.csv
  test_<mode>.csv

Usage Examples:
  python balance_dataset.py --input combined_marathi_dataset.csv --mode strict-domain
  python balance_dataset.py --input combined_marathi_dataset.csv --mode label-only --seed 123
  python balance_dataset.py --input combined_marathi_dataset.csv --mode proportional

Requirements: pandas, scikit-learn
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

VALID_MODES = {"strict-domain", "label-only", "proportional"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    expected = {"text", "label", "source"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    # Basic cleaning
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def balance_strict_domain(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = []
    for label, sub_label in df.groupby("label"):
        per_source_counts = sub_label["source"].value_counts()
        if per_source_counts.empty:
            continue
        min_count = per_source_counts.min()
        if min_count <= 0:
            continue
        for source_val, sub_src in sub_label.groupby("source"):
            take = min(min_count, len(sub_src))
            groups.append(sub_src.sample(take, random_state=seed, replace=False))
    if not groups:
        return df.copy()
    return pd.concat(groups, ignore_index=True)


def balance_label_only(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    if counts.empty:
        return df.copy()
    min_count = counts.min()
    sampled = []
    for label, sub in df.groupby("label"):
        take = min(min_count, len(sub))
        sampled.append(sub.sample(take, random_state=seed, replace=False))
    return pd.concat(sampled, ignore_index=True)


def perform_split(df: pd.DataFrame, mode: str, seed: int):
    if mode == "proportional":
        return None, None, None  # no splits for proportional (acts as baseline)

    if mode == "strict-domain":
        strat_key = df["label"].astype(str) + "|" + df["source"].astype(str)
    else:  # label-only
        strat_key = df["label"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=strat_key
    )
    strat_temp = (
        temp_df["label"].astype(str) + "|" + temp_df["source"].astype(str)
        if mode == "strict-domain" else temp_df["label"].astype(str)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=strat_temp
    )
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser(description="Balance Marathi sentiment dataset.")
    ap.add_argument("--input", required=True, help="Path to combined dataset CSV")
    ap.add_argument("--mode", required=True, choices=VALID_MODES, help="Balancing mode")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--output-dir", default=None, help="Override output directory (default: same as input)")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print("Label distribution (original):")
    print(df['label'].value_counts())
    print("Source distribution (original):")
    if 'source' in df.columns:
        print(df['source'].value_counts())

    if args.mode == "strict-domain":
        balanced = balance_strict_domain(df, seed=args.seed)
    elif args.mode == "label-only":
        balanced = balance_label_only(df, seed=args.seed)
    else:  # proportional
        balanced = df.copy()

    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"Balanced size: {len(balanced)}")
    print("Balanced label distribution:")
    print(balanced['label'].value_counts())
    if 'source' in balanced.columns:
        print("Balanced source distribution:")
        print(balanced['source'].value_counts())

    mode_tag = args.mode.replace('-', '_')
    balanced_path = out_dir / f"balanced_mode_{mode_tag}.csv"
    balanced.to_csv(balanced_path, index=False, encoding='utf-8')
    print(f"Saved balanced dataset: {balanced_path}")

    train_df = val_df = test_df = None
    if args.mode != "proportional":
        train_df, val_df, test_df = perform_split(balanced, args.mode, args.seed)
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out_path = out_dir / f"{name}_{mode_tag}.csv"
            part.to_csv(out_path, index=False, encoding='utf-8')
            print(f"Saved {name} split: {out_path} (rows={len(part)})")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
