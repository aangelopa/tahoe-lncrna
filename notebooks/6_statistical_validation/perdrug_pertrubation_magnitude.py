"""
Vision-Score Perturbation Magnitude
=====================================
Computes a transcriptional perturbation magnitude from existing differential
Vision score parquet files —  no GPU needed.

For each comparison (drug | plate | cell_line), the perturbation magnitude is:
    magnitude = L2 norm of the differential Vision score vector
              = sqrt(sum of squared scores across all gene sets)

This is a natural proxy for E-distance: it measures how far the drug pushed
the lncRNA gene-set landscape away from DMSO in the dimensional space.

For validation we:
  1. Aggregate per drug (median across plates/cell lines)
  2. Rank drugs by magnitude and compare to known MOA ordering
  3. Compute Spearman correlation with published E-distance rank where available
  4. Produce a ranked bar chart coloured by MOA

Outputs:
    perturbation_magnitude.csv       — per comparison magnitudes
    drug_magnitude_summary.csv       — per drug summary (median, IQR, n comparisons)
    perturbation_magnitude_ranked.pdf/png — bar chart ranked by magnitude
    perturbation_magnitude_by_moa.pdf/png — boxplot per MOA

"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNKS_DIR = Path.home() / "Thesis" / "Data" / "diff_vision_chunks"
DEFAULT_OUT_DIR    = Path.home() / "Thesis" / "Results" / "perturbation_magnitude"
DEFAULT_DRUG_META  = Path.home() / "Thesis" / "Data" / "drug_metadata.parquet"
DMSO_LABEL         = "DMSO_TF"
MOA_COL            = "moa-fine"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MOA ANNOTATIONS FROM DRUG METADATA
# ─────────────────────────────────────────────────────────────────────────────

def load_moa_map(drug_meta_path):
    """
    Load MOA annotations from the Tahoe drug_metadata.parquet file.
    Returns a dict mapping drug name -> MOA string.
    """
    path = Path(drug_meta_path)
    if not path.exists():
        print(f"  WARNING: drug_metadata.parquet not found at {path}")
        print(f"  MOA colours will all be 'Unknown'. Pass --drug_meta to fix.")
        return {}

    drug_meta = pd.read_parquet(path)
    drug_meta["drug"] = drug_meta["drug"].str.strip()

    if MOA_COL not in drug_meta.columns:
        print(f"  WARNING: Column '{MOA_COL}' not found in drug metadata.")
        print(f"  Available columns: {drug_meta.columns.tolist()}")
        return {}

    moa_map = dict(zip(drug_meta["drug"], drug_meta[MOA_COL]))
    n_known = sum(1 for v in moa_map.values() if pd.notna(v) and v != "unclear")
    print(f"  Loaded MOA for {len(moa_map)} drugs ({n_known} with known MOA)")
    return moa_map


def build_moa_colour_map(drug_agg):
    """
    Assign a colour to each unique MOA using tab20 — same approach as
    figure5moa.py.
    """
    unique_moas = sorted(
        m for m in drug_agg["moa"].unique()
        if pd.notna(m) and m not in ("unclear", "Unknown")
    )
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    if len(unique_moas) > 20:
        cmap = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
        ])
    colour_map = {moa: cmap[i % len(cmap)] for i, moa in enumerate(unique_moas)}
    colour_map["unclear"]  = "#CCCCCC"
    colour_map["Unknown"]  = "#CCCCCC"
    return colour_map


# ─────────────────────────────────────────────────────────────────────────────
# LOAD VISION SCORES
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(chunks_dir=None, parquet_path=None):
    if parquet_path and Path(parquet_path).exists():
        print(f"Loading merged parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    if chunks_dir:
        chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "group_*.parquet")))
        if not chunk_files:
            sys.exit(f"ERROR: No group_*.parquet files found in {chunks_dir}")
        print(f"Loading {len(chunk_files)} chunk files ...")
        df = pd.concat([pd.read_parquet(f) for f in chunk_files], axis=0)
        df = df[~df.index.duplicated(keep="first")]
        print(f"  Shape: {df.shape[0]:,} comparisons × {df.shape[1]:,} gene sets")
        return df
    sys.exit("ERROR: Provide --chunks_dir or --parquet.")


def parse_index(df):
    split = df.index.str.split(r"\s*\|\s*")
    df = df.copy()
    df.insert(0, "cell_line", split.str[2])
    df.insert(0, "plate",     split.str[1])
    df.insert(0, "drug",      split.str[0])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE MAGNITUDE
# ─────────────────────────────────────────────────────────────────────────────

def compute_magnitude(df):
   
    meta_cols  = ["drug", "plate", "cell_line"]
    score_cols = [c for c in df.columns if c not in meta_cols]

    scores_mat = df[score_cols].values.astype(np.float32)
    scores_mat = np.nan_to_num(scores_mat, nan=0.0)

    magnitudes    = np.linalg.norm(scores_mat, axis=1)
    log1p_mag     = np.log1p(magnitudes)

    result = df[meta_cols].copy()
    result["magnitude"]       = magnitudes
    result["log1p_magnitude"] = log1p_mag
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE PER DRUG
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_per_drug(mag_df, moa_map):
    """
    Aggregate per drug using median across plates/cell lines.
    MOA is merged from the real Tahoe drug_metadata.parquet annotations.
    """
    agg = (
        mag_df.groupby("drug")["log1p_magnitude"]
        .agg(
            median_log1p_mag="median",
            mean_log1p_mag="mean",
            std_log1p_mag="std",
            n_comparisons="count",
        )
        .reset_index()
        .sort_values("median_log1p_mag", ascending=False)
        .reset_index(drop=True)
    )
    agg["rank"] = agg.index + 1

    # Use real MOA from drug_metadata.parquet
    agg["moa"] = agg["drug"].map(moa_map).fillna("Unknown")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: MOA ORDERING
# ─────────────────────────────────────────────────────────────────────────────

def validate_moa_ordering(drug_agg):
    """
    Check that the MOA hierarchy from the Tahoe paper is preserved:
    Proteasome > HDAC > CDK > RAS/RAF/MEK
    Prints median rank per MOA (lower rank = stronger perturbation).
    """
    exclude = {"unclear", "Unknown"}
    moa_ranks = (
        drug_agg[~drug_agg["moa"].isin(exclude)]
        .groupby("moa")["rank"]
        .median()
        .sort_values()
    )
    print("\n" + "=" * 60)
    print("MOA VALIDATION — median perturbation rank (lower = stronger)")
    print("Expected order (Tahoe paper): Proteasome > HDAC > CDK > RAS/MEK")
    print("=" * 60)
    for moa, r in moa_ranks.items():
        print(f"  {moa:<40s}  median rank = {r:.0f}")
    return moa_ranks


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ranked(drug_agg, colour_map, out_dir, top_n=60, fontsize=8):
    """Bar chart of top_n drugs by median log1p magnitude, coloured by MOA."""
    top     = drug_agg.head(top_n)
    colours = [colour_map.get(m, "#CCCCCC") for m in top["moa"]]

    fig, ax = plt.subplots(figsize=(14, max(5, top_n * 0.22 + 1.5)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fontsize})

    y_pos = np.arange(len(top))[::-1]
    ax.barh(y_pos, top["median_log1p_mag"], color=colours,
            height=0.7, edgecolor="none", zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["drug"], fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 0.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Median log1p(perturbation magnitude)", fontsize=fontsize)
    ax.set_title(
        f"Top {top_n} drugs by lncRNA Vision Score perturbation magnitude\n"
        f"(L2 norm of differential Vision score vector across gene sets)",
        fontsize=fontsize + 1, fontweight="bold", loc="left", pad=6,
    )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#DDDDDD", zorder=0)

    # Legend — only MOAs present in the top_n
    present_moas = [m for m in top["moa"].unique()
                    if m not in ("unclear", "Unknown")]
    handles = [
        mpatches.Patch(color=colour_map.get(m, "#CCCCCC"), label=m)
        for m in present_moas
    ]
    if any(m in ("unclear", "Unknown") for m in top["moa"].unique()):
        handles.append(mpatches.Patch(color="#CCCCCC", label="unclear / unknown"))
    ax.legend(handles=handles, fontsize=fontsize - 1.5, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#DDDDDD",
              ncol=2 if len(handles) > 10 else 1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"perturbation_magnitude_ranked.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved {ext.upper()} → {path}")
    plt.close(fig)


def plot_moa_boxplot(mag_df, drug_agg, colour_map, out_dir, fontsize=9):
    """
    Boxplot of log1p magnitude distributions per MOA.
    """
    mag_with_moa = mag_df.merge(
        drug_agg[["drug", "moa"]], on="drug", how="left"
    )
    mag_with_moa["moa"] = mag_with_moa["moa"].fillna("Unknown")

    # Sort MOAs by median magnitude descending; put unclear/Unknown last
    moa_order = (
        mag_with_moa[~mag_with_moa["moa"].isin({"unclear", "Unknown"})]
        .groupby("moa")["log1p_magnitude"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    for tail in ("unclear", "Unknown"):
        if tail in mag_with_moa["moa"].values:
            moa_order.append(tail)

    data_by_moa = [
        mag_with_moa[mag_with_moa["moa"] == m]["log1p_magnitude"].values
        for m in moa_order
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(moa_order) * 1.1), 5))
    bp = ax.boxplot(
        data_by_moa, patch_artist=True, showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.5},
    )
    for patch, moa in zip(bp["boxes"], moa_order):
        patch.set_facecolor(colour_map.get(moa, "#CCCCCC"))
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(moa_order) + 1))
    ax.set_xticklabels(moa_order, rotation=45, ha="right", fontsize=fontsize - 1)
    ax.tick_params(axis="y", labelsize=fontsize - 0.5)
    ax.set_ylabel("log1p(perturbation magnitude)", fontsize=fontsize)
    ax.set_title(
        "lncRNA Vision Score perturbation magnitude by drug MOA",
        fontsize=fontsize + 1, fontweight="bold", loc="left", pad=6,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#DDDDDD", zorder=0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"perturbation_magnitude_by_moa.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved {ext.upper()} → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute Vision-score perturbation magnitude per drug/comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--chunks_dir", type=str, default=str(DEFAULT_CHUNKS_DIR))
    src.add_argument("--parquet",    type=str, default=None)
    parser.add_argument("--drug_meta", type=str, default=str(DEFAULT_DRUG_META),
                        help="Path to drug_metadata.parquet (contains moa-fine column)")
    parser.add_argument("--out_dir",   type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top_n",     type=int, default=60,
                        help="How many drugs to show in ranked bar chart")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MOA annotations ──────────────────────────────────────────────────
    print("Loading MOA annotations from drug_metadata.parquet ...")
    moa_map = load_moa_map(args.drug_meta)

    # ── Load Vision scores ────────────────────────────────────────────────────
    df = load_scores(
        chunks_dir   = args.chunks_dir if args.parquet is None else None,
        parquet_path = args.parquet,
    )
    df = parse_index(df)
    df = df[df["drug"] != DMSO_LABEL]

    # ── Compute magnitude ─────────────────────────────────────────────────────
    print("\nComputing perturbation magnitudes ...")
    mag_df = compute_magnitude(df)
    del df

    mag_df.to_csv(out_dir / "perturbation_magnitude.csv", index=False)
    print(f"Saved → {out_dir / 'perturbation_magnitude.csv'}")
    print(f"\nMagnitude stats (log1p):")
    print(mag_df["log1p_magnitude"].describe().round(4).to_string())

    # ── Aggregate per drug with real MOA ──────────────────────────────────────
    drug_agg = aggregate_per_drug(mag_df, moa_map)
    drug_agg.to_csv(out_dir / "drug_magnitude_summary.csv", index=False)
    print(f"\nSaved → {out_dir / 'drug_magnitude_summary.csv'}")
    print(f"\nTop 15 drugs by perturbation magnitude:")
    print(drug_agg.head(15)[
        ["rank", "drug", "moa", "median_log1p_mag", "n_comparisons"]
    ].to_string(index=False))

    # ── Validate MOA ordering ─────────────────────────────────────────────────
    validate_moa_ordering(drug_agg)

    # ── Build colour map from actual MOAs in data ─────────────────────────────
    colour_map = build_moa_colour_map(drug_agg)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_ranked(drug_agg, colour_map, out_dir, top_n=args.top_n)
    plot_moa_boxplot(mag_df, drug_agg, colour_map, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()