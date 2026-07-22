"""
Hierarchical Clustering of Differential Vision Scores
=====================================================
Clusters drugs by their lncRNA gene set perturbation profiles.

Method:
  1. Load differential Vision score parquets
  2. Aggregate by drug (median across cell lines and plates)
  3. Hierarchical clustering with correlation distance + Ward linkage
  4. Clustermap with MOA row color annotation

Output:
  clustermap_drugs_genesets.pdf
  drug_median_scores.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from pathlib import Path
import os
import glob
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR = Path.home() / "Thesis"
DATA_DIR = BASE_DIR / "Data"
PARQUET_DIR = DATA_DIR  # where group_*.parquet files are
DRUG_META_PATH = DATA_DIR / "drug_metadata.parquet"
OUTPUT_DIR = BASE_DIR / "Results" / "hierarchical_clustering"

MOA_COL = "moa-fine"
DISTANCE_METRIC = "correlation"  # 1 - Pearson correlation
LINKAGE_METHOD = "ward"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DIFFERENTIAL VISION SCORES
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading differential Vision scores")
print("=" * 60)
t0 = time.time()

parquet_files = sorted(glob.glob(str(PARQUET_DIR / "group_*.parquet")))
print(f"  Found {len(parquet_files)} parquet files")

scores_df = pd.concat([pd.read_parquet(f) for f in parquet_files], axis=0)
print(f"  Merged shape: {scores_df.shape}")

# Parse metadata from index: 'drug | plate | cell_line'
parts = scores_df.index.str.rsplit(" | ", n=2)
scores_df["drug"] = [p[0] for p in parts]
scores_df["plate"] = [p[1] for p in parts]
scores_df["cell_line"] = [p[2] for p in parts]

print(f"  Unique drugs: {scores_df['drug'].nunique()}")
print(f"  Unique plates: {scores_df['plate'].nunique()}")
print(f"  Unique cell lines: {scores_df['cell_line'].nunique()}")

# ─────────────────────────────────────────────
# 2. AGGREGATE BY DRUG (median across cell lines + plates)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Aggregating by drug (median)")
print("=" * 60)

# Get gene set columns (everything except drug, plate, cell_line)
geneset_cols = [c for c in scores_df.columns if c not in ["drug", "plate", "cell_line"]]
print(f"  Gene sets: {len(geneset_cols)}")

drug_median = scores_df.groupby("drug")[geneset_cols].median()
print(f"  Drug median matrix: {drug_median.shape}")

# Remove any drugs with all NaN
drug_median = drug_median.dropna(how="all")
# Fill remaining NaN with 0
drug_median = drug_median.fillna(0)
print(f"  After cleanup: {drug_median.shape}")

# Save for reuse
drug_median.to_parquet(OUTPUT_DIR / "drug_median_scores.parquet")
print(f"  Saved → drug_median_scores.parquet")

# ─────────────────────────────────────────────
# 3. LOAD MOA ANNOTATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Loading MOA annotations")
print("=" * 60)

drug_meta = pd.read_parquet(DRUG_META_PATH)
drug_meta["drug"] = drug_meta["drug"].str.strip()

# Create MOA lookup
moa_lookup = drug_meta.set_index("drug")[MOA_COL].to_dict()

# Map MOA to drugs in our matrix
drug_moa = pd.Series(
    [moa_lookup.get(d, "unknown") for d in drug_median.index],
    index=drug_median.index,
    name="MOA"
)

# Replace NaN with "unknown"
drug_moa = drug_moa.fillna("unknown")

# Stats
moa_counts = drug_moa.value_counts()
print(f"  MOA categories: {len(moa_counts)}")
for moa, count in moa_counts.head(10).items():
    print(f"    {moa}: {count}")

# ─────────────────────────────────────────────
# 4. BUILD MOA COLOR MAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Building color maps")
print("=" * 60)

unique_moas = sorted(drug_moa.unique())
n_moas = len(unique_moas)

# Color palette for MOAs
all_colors = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
])

moa_color_map = {}
for i, moa in enumerate(unique_moas):
    if moa in ["unclear", "unknown"]:
        moa_color_map[moa] = (0.85, 0.85, 0.85, 1.0)  # light gray
    else:
        moa_color_map[moa] = tuple(all_colors[i % len(all_colors)])

# Create row colors DataFrame
row_colors = drug_moa.map(moa_color_map)
row_colors.name = "MOA"

print(f"  {n_moas} MOA categories mapped to colors")

# ─────────────────────────────────────────────
# 5. HIERARCHICAL CLUSTERING (CLUSTERMAP)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Hierarchical clustering")
print("=" * 60)

print(f"  Distance metric: {DISTANCE_METRIC}")
print(f"  Linkage method: {LINKAGE_METHOD}")
print(f"  Matrix: {drug_median.shape[0]} drugs × {drug_median.shape[1]} gene sets")

# Clamp extreme values for better visualization
vmax = np.percentile(drug_median.values, 99)
vmin = np.percentile(drug_median.values, 1)

g = sns.clustermap(
    drug_median,
    method=LINKAGE_METHOD,
    metric=DISTANCE_METRIC,
    row_colors=row_colors,
    cmap="RdBu_r",
    center=0,
    vmin=vmin,
    vmax=vmax,
    figsize=(20, max(12, drug_median.shape[0] * 0.15)),
    xticklabels=False,  # too many gene sets to label
    yticklabels=True,
    dendrogram_ratio=(0.15, 0.05),
    colors_ratio=0.02,
    cbar_kws={"label": "Differential Vision Score", "shrink": 0.5},
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
)

# Fix drug name font size
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=4)
g.ax_heatmap.set_xlabel("Gene Sets", fontsize=10)
g.ax_heatmap.set_ylabel("")

# Add MOA legend
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor=moa_color_map[moa], label=moa)
    for moa in unique_moas if moa not in ["unclear", "unknown"]
]
# Add unclear/unknown at the end
for moa in ["unclear", "unknown"]:
    if moa in unique_moas:
        legend_patches.append(Patch(facecolor=moa_color_map[moa], label=moa))

g.ax_heatmap.legend(
    handles=legend_patches,
    title="Mechanism of Action",
    bbox_to_anchor=(1.25, 1),
    loc="upper left",
    fontsize=5,
    title_fontsize=7,
    framealpha=0.9,
    ncol=1,
)

g.fig.suptitle(
    f"Hierarchical clustering of drugs by differential Vision scores\n"
    f"({drug_median.shape[0]} drugs × {drug_median.shape[1]} gene sets, "
    f"{DISTANCE_METRIC} distance, {LINKAGE_METHOD} linkage)",
    fontsize=12, y=1.02
)

out_path = OUTPUT_DIR / "clustermap_drugs_genesets.pdf"
g.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

# ─────────────────────────────────────────────
# 6. SMALLER CLUSTERMAP (exclude unclear/unknown MOA)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Clustermap for known MOA drugs only")
print("=" * 60)

known_mask = ~drug_moa.isin(["unclear", "unknown"])
drug_median_known = drug_median[known_mask]
row_colors_known = row_colors[known_mask]

print(f"  Known MOA drugs: {drug_median_known.shape[0]}")

if drug_median_known.shape[0] > 5:
    vmax_k = np.percentile(drug_median_known.values, 99)
    vmin_k = np.percentile(drug_median_known.values, 1)

    g2 = sns.clustermap(
        drug_median_known,
        method=LINKAGE_METHOD,
        metric=DISTANCE_METRIC,
        row_colors=row_colors_known,
        cmap="RdBu_r",
        center=0,
        vmin=vmin_k,
        vmax=vmax_k,
        figsize=(18, max(10, drug_median_known.shape[0] * 0.2)),
        xticklabels=False,
        yticklabels=True,
        dendrogram_ratio=(0.15, 0.05),
        colors_ratio=0.02,
        cbar_kws={"label": "Differential Vision Score", "shrink": 0.5},
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )

    g2.ax_heatmap.set_yticklabels(g2.ax_heatmap.get_yticklabels(), fontsize=5)
    g2.ax_heatmap.set_xlabel("Gene Sets", fontsize=10)
    g2.ax_heatmap.set_ylabel("")

    # MOA legend (only known)
    known_moas = sorted(drug_moa[known_mask].unique())
    legend_patches_known = [
        Patch(facecolor=moa_color_map[moa], label=moa)
        for moa in known_moas
    ]
    g2.ax_heatmap.legend(
        handles=legend_patches_known,
        title="Mechanism of Action",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=8,
        framealpha=0.9,
        ncol=1,
    )

    g2.fig.suptitle(
        f"Hierarchical clustering — Known MOA drugs only\n"
        f"({drug_median_known.shape[0]} drugs × {drug_median_known.shape[1]} gene sets)",
        fontsize=12, y=1.02
    )

    out_path2 = OUTPUT_DIR / "clustermap_drugs_known_moa.pdf"
    g2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path2}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'=' * 60}")
print("DONE")
print(f"  Total drugs:          {drug_median.shape[0]}")
print(f"  Known MOA drugs:      {sum(known_mask)}")
print(f"  Gene sets:            {drug_median.shape[1]}")
print(f"  Distance:             {DISTANCE_METRIC}")
print(f"  Linkage:              {LINKAGE_METHOD}")
print(f"  Total time:           {total:.0f}s")
print(f"\nOutputs in {OUTPUT_DIR}:")
print(f"  clustermap_drugs_genesets.pdf    (all drugs)")
print(f"  clustermap_drugs_known_moa.pdf   (known MOA only)")
print(f"  drug_median_scores.parquet       (aggregated matrix)")
print(f"{'=' * 60}")