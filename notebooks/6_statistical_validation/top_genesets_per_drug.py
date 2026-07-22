"""
Top Gene Sets Per Drug — from Differential Vision Scores
=========================================================
For each drug, aggregates differential Vision scores across all
(plate, cell_line) comparisons and reports the 10 most overexpressed
and 10 most underexpressed gene sets.

Aggregation strategy (default: median across comparisons):
  - Median is robust to outlier cell lines / plates.
  

Input:
  - Directory of group_XXXX.parquet files  (--chunks_dir), OR
  - A single merged parquet file           (--parquet)

Output:
  - CSV:  top_genesets_per_drug.csv   (long format, one row per drug × geneset)
  - TSV:  top_genesets_per_drug.tsv   (same, tab-separated)
  - TXT:  top_genesets_per_drug.txt   (human-readable summary)

Index format in parquet:  "drug | plate | cell_line"


"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNKS_DIR  = Path.home() / "Thesis" / "Data" / "diff_vision_chunks"
DEFAULT_PARQUET     = Path.home() / "Thesis" / "Data" / "differential_vision_scores.parquet"
DEFAULT_OUT_DIR     = Path.home() / "Thesis" / "Results" / "top_genesets"
DEFAULT_N           = 10
DEFAULT_AGG         = "median"   # "median" | "mean"
DMSO_LABEL          = "DMSO_TF"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(chunks_dir=None, parquet_path=None):
    """
    Load differential vision scores into a DataFrame.
    Index: "drug | plate | cell_line"
    Columns: gene set names
    """
    if parquet_path and Path(parquet_path).exists():
        print(f"Loading merged parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"  Shape: {df.shape[0]:,} comparisons × {df.shape[1]:,} gene sets")
        return df

    if chunks_dir:
        chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "group_*.parquet")))
        if not chunk_files:
            sys.exit(f"ERROR: No group_*.parquet files found in {chunks_dir}")
        print(f"Loading {len(chunk_files)} chunk files from {chunks_dir} ...")
        dfs = [pd.read_parquet(f) for f in chunk_files]
        df  = pd.concat(dfs, axis=0)
        # Drop exact duplicate indices if any (shouldn't happen, but safe)
        n_dups = df.index.duplicated().sum()
        if n_dups:
            print(f"  WARNING: dropping {n_dups} duplicate rows")
            df = df[~df.index.duplicated(keep="first")]
        print(f"  Shape: {df.shape[0]:,} comparisons × {df.shape[1]:,} gene sets")
        return df

    sys.exit("ERROR: Provide --chunks_dir or --parquet.")


# ─────────────────────────────────────────────────────────────────────────────
# PARSE INDEX → drug / plate / cell_line COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def parse_index(df):
    """
    Split index strings of the form  "drug | plate | cell_line"
    into three metadata columns and return a copy.
    """
    split = df.index.str.split(r"\s*\|\s*", expand=True)
    if split.shape[1] < 3:
        sys.exit(
            "ERROR: Index does not match expected format 'drug | plate | cell_line'.\n"
            f"  Example row: {df.index[0]!r}"
        )
    df = df.copy()
    df.insert(0, "cell_line", split[2])
    df.insert(0, "plate",     split[1])
    df.insert(0, "drug",      split[0])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE SCORES PER DRUG
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_per_drug(df, agg="median"):
    """
    df must have a 'drug' column + gene-set score columns.
    Returns a DataFrame: rows = drugs, cols = gene sets.
    """
    geneset_cols = [c for c in df.columns if c not in ("drug", "plate", "cell_line")]
    agg_fn       = np.median if agg == "median" else np.mean
    print(f"\nAggregating scores per drug using {agg} across comparisons ...")

    grouped = df.groupby("drug", observed=True)[geneset_cols]
    if agg == "median":
        drug_scores = grouped.median()
    else:
        drug_scores = grouped.mean()

    print(f"  {len(drug_scores):,} unique drugs found.")
    return drug_scores


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT TOP N OVER / UNDER EXPRESSED
# ─────────────────────────────────────────────────────────────────────────────

def get_top_n(drug_scores, n=10):
    """
    For each drug, return the top-n over- and under-expressed gene sets.
    Returns a long-format DataFrame.
    """
    records = []
    for drug in drug_scores.index:
        row  = drug_scores.loc[drug]
        row  = row.dropna()

        # Top N highest (overexpressed)
        top_over  = row.nlargest(n)
        for rank, (gs, score) in enumerate(top_over.items(), start=1):
            records.append({
                "drug":      drug,
                "direction": "overexpressed",
                "rank":      rank,
                "gene_set":  gs,
                "score":     round(float(score), 6),
            })

        # Top N lowest (underexpressed)
        top_under = row.nsmallest(n)
        for rank, (gs, score) in enumerate(top_under.items(), start=1):
            records.append({
                "drug":      drug,
                "direction": "underexpressed",
                "rank":      rank,
                "gene_set":  gs,
                "score":     round(float(score), 6),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results_df, drug_scores, out_dir, n, agg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV (long format) ─────────────────────────────────────────────────────
    csv_path = out_dir / "top_genesets_per_drug.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV  → {csv_path}")

    # ── TSV ───────────────────────────────────────────────────────────────────
    tsv_path = out_dir / "top_genesets_per_drug.tsv"
    results_df.to_csv(tsv_path, index=False, sep="\t")
    print(f"Saved TSV  → {tsv_path}")

    # ── Human-readable TXT ────────────────────────────────────────────────────
    txt_path = out_dir / "top_genesets_per_drug.txt"
    drugs    = results_df["drug"].unique()

    with open(txt_path, "w") as f:
        f.write(
            f"Top {n} Over- and Under-expressed Gene Sets per Drug\n"
            f"Aggregation: {agg} across all (plate, cell_line) comparisons\n"
            f"Total drugs: {len(drugs)}\n"
            f"{'=' * 80}\n\n"
        )
        for drug in sorted(drugs):
            sub = results_df[results_df["drug"] == drug]
            over  = sub[sub["direction"] == "overexpressed"].sort_values("rank")
            under = sub[sub["direction"] == "underexpressed"].sort_values("rank")

            n_comp = drug_scores.loc[drug].dropna(how="all").shape[0] \
                if drug in drug_scores.index else "?"
            # number of comparisons contributing (before aggregation)
            f.write(f"DRUG: {drug}\n")
            f.write(f"{'-' * 60}\n")

            f.write(f"  TOP {n} OVEREXPRESSED (highest diff Vision scores):\n")
            for _, row in over.iterrows():
                f.write(f"    {row['rank']:>2}. {row['gene_set']:<60s}  score={row['score']:+.4f}\n")

            f.write(f"\n  TOP {n} UNDEREXPRESSED (lowest diff Vision scores):\n")
            for _, row in under.iterrows():
                f.write(f"    {row['rank']:>2}. {row['gene_set']:<60s}  score={row['score']:+.4f}\n")

            f.write("\n")

    print(f"Saved TXT  → {txt_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Drugs processed     : {len(drugs):,}")
    print(f"  Gene sets evaluated : {drug_scores.shape[1]:,}")
    print(f"  Aggregation method  : {agg}")
    print(f"  Top N per direction : {n}")
    print(f"  Output directory    : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK PREVIEW (printed to stdout)
# ─────────────────────────────────────────────────────────────────────────────

def preview(results_df, n_drugs=5, n=10):
    """Print a preview for the first few drugs."""
    drugs = results_df["drug"].unique()[:n_drugs]
    print(f"\n{'=' * 80}")
    print(f"PREVIEW — first {n_drugs} drugs")
    print(f"{'=' * 80}")
    for drug in drugs:
        sub   = results_df[results_df["drug"] == drug]
        over  = sub[sub["direction"] == "overexpressed"].sort_values("rank")
        under = sub[sub["direction"] == "underexpressed"].sort_values("rank")
        print(f"\n  {drug}")
        print(f"  {'─' * 70}")
        print(f"  OVER  : " + " | ".join(
            f"{r['gene_set']} ({r['score']:+.3f})"
            for _, r in over.head(3).iterrows()
        ) + " ...")
        print(f"  UNDER : " + " | ".join(
            f"{r['gene_set']} ({r['score']:+.3f})"
            for _, r in under.head(3).iterrows()
        ) + " ...")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Top N over/under-expressed gene sets per drug from diff Vision scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--chunks_dir", type=str,
        default=str(DEFAULT_CHUNKS_DIR),
        help=f"Directory with group_*.parquet files (default: {DEFAULT_CHUNKS_DIR})",
    )
    src.add_argument(
        "--parquet", type=str,
        default=None,
        help="Path to a single merged parquet file",
    )

    # Options
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Number of top gene sets per direction (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--agg", choices=["median", "mean"], default=DEFAULT_AGG,
        help=f"Aggregation across comparisons (default: {DEFAULT_AGG})",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--drugs", nargs="+", default=None,
        help="Optional: restrict analysis to specific drug names",
    )
    parser.add_argument(
        "--exclude_dmso", action="store_true", default=True,
        help="Exclude DMSO_TF rows from analysis (default: True)",
    )
    parser.add_argument(
        "--no_preview", action="store_true",
        help="Skip stdout preview",
    )

    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_scores(
        chunks_dir  = args.chunks_dir if args.parquet is None else None,
        parquet_path= args.parquet,
    )

    # ── Parse index ───────────────────────────────────────────────────────────
    df = parse_index(df)

    # ── Filter ────────────────────────────────────────────────────────────────
    if args.exclude_dmso:
        before = len(df)
        df     = df[df["drug"] != DMSO_LABEL]
        print(f"  Excluded DMSO rows: {before - len(df):,} removed, {len(df):,} remain")

    if args.drugs:
        df = df[df["drug"].isin(args.drugs)]
        if df.empty:
            sys.exit(f"ERROR: None of the requested drugs found: {args.drugs}")
        print(f"  Filtered to {len(df['drug'].unique())} requested drugs.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    drug_scores = aggregate_per_drug(df, agg=args.agg)

    # ── Top N ─────────────────────────────────────────────────────────────────
    print(f"Computing top {args.n} over/under-expressed gene sets per drug ...")
    results_df = get_top_n(drug_scores, n=args.n)

    # ── Preview ───────────────────────────────────────────────────────────────
    if not args.no_preview:
        preview(results_df, n_drugs=5, n=args.n)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(results_df, drug_scores, args.out_dir, args.n, args.agg)

    print("\nDone.")


if __name__ == "__main__":
    main()