"""
Comparison-Level Leiden — Follow-Up Checks 
========================================================
Builds on comparison_leiden_metadata.parquet (from comparison_level_full.py
/ resume_from_step6.py). Adds three quantitative checks discussed:

  [CHECK 1] What fraction of comparisons fall into the small, cell-line-
  dominated "residual" clusters (8-15 in your run)? Turns a visual
  impression from the heatmap into a concrete number/caveat.

  [CHECK 2] Quantify MOA-clustering strength, not just eyeball the heatmap:
    - Adjusted Rand Index (ARI) between Leiden clusters and MOA labels
      (measures overall agreement across ALL clusters/MOAs at once)
    - Hypergeometric enrichment test for your key cluster (7, the
      HDAC/proteasome-dominated one) against its top MOAs specifically
      (a sharper, more targeted significance test for the one finding
      you actually want to report)

  [CHECK 3] ONE targeted higher-resolution rerun (not a blind sweep) -
  tests whether cluster 7 (HDAC/proteasome/CDK/PI3K-AKT/protein-synthesis)
  stays intact as a single cluster, or fragments into sub-signatures, at
  a higher resolution. Mirrors the same targeted approach used for the
  drug-level pipeline's cluster 0/2 split.

Requires: comparison_leiden_metadata.parquet, plus the PCs used for Leiden
(regenerated here cheaply from saved z-score params + raw parquet chunks,
same approach as resume_from_step6.py — avoids needing the expensive
t-SNE/NNMDS steps again).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import glob
import os
import time
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION — must match your comparison_level_full.py run
# ─────────────────────────────────────────────
DATA_DIR   = "/home/a/aangelopa/Thesis/Data/diff_vision_chunks"
OUTPUT_DIR = "/home/a/aangelopa/Thesis/Data/2ndtry/comparison_level_full"

MOA_COL       = "moa-fine"
EXCLUDE_PLATE = "14"

N_PCS_LEIDEN      = 50
LEIDEN_RESOLUTION = 0.5     # original resolution
RESOLUTION_HIGH   = 1.0     # targeted rerun for check 5
N_NEIGHBORS       = 15
RANDOM_STATE      = 42

# The cluster you want to track through check 5 - update if your own
# cluster numbering differs (this was "cluster 7" in your uploaded heatmaps,
# dominated by HDAC inhibitor / Proteasome inhibitor)
KEY_CLUSTER = "7"

USE_GPU_PCA = False   # CPU sklearn - avoids the Ampere queue for this check

META_PATH = os.path.join(OUTPUT_DIR, "comparison_leiden_metadata.parquet")

# ─────────────────────────────────────────────
# LOAD EXISTING METADATA
# ─────────────────────────────────────────────
print("Loading existing comparison-level metadata...")
meta = pd.read_parquet(META_PATH)
print(f"  {len(meta):,} comparisons, {meta['leiden_comp'].nunique()} clusters")

# ═════════════════════════════════════════════
# CHECK 1: fraction of comparisons in small, cell-line-dominated clusters
# ═════════════════════════════════════════════
print("\n" + "=" * 60)
print("CHECK 2: Residual cell-line-dominated clusters")
print("=" * 60)

cluster_sizes = meta["leiden_comp"].value_counts().sort_index()
print("\nCluster sizes:")
print(cluster_sizes)

# Identify "residual" clusters automatically: for each cluster, what
# fraction of its comparisons come from its single most common cell line?
# A cluster that's mostly one cell line is a candidate "residual" cluster.
purity = (
    meta.groupby("leiden_comp")["cell_line"]
    .apply(lambda s: s.value_counts(normalize=True).iloc[0])
)
print("\nCluster purity (fraction of cluster from its single most common cell line):")
print(purity.sort_values(ascending=False))

RESIDUAL_PURITY_THRESHOLD = 0.5   # >50% from one cell line = "residual"
residual_clusters = purity[purity > RESIDUAL_PURITY_THRESHOLD].index.tolist()
n_residual = meta["leiden_comp"].isin(residual_clusters).sum()
frac_residual = n_residual / len(meta)

print(f"\nClusters exceeding {RESIDUAL_PURITY_THRESHOLD:.0%} single-cell-line purity: "
      f"{residual_clusters}")
print(f"Comparisons in these residual clusters: {n_residual:,} / {len(meta):,} "
      f"({frac_residual:.2%})")

if frac_residual < 0.05:
    print("-> Small fraction (<5%) - a brief one-sentence caveat is sufficient.")
elif frac_residual < 0.15:
    print("-> Moderate fraction (5-15%) - worth reporting explicitly with the number.")
else:
    print("-> Substantial fraction (>15%) - cell-line confound may still be")
    print("   meaningfully affecting your overall result, not just a few outliers.")

# ═════════════════════════════════════════════
# CHECK 2: quantify MOA-clustering strength
# ═════════════════════════════════════════════
print("\n" + "=" * 60)
print("CHECK 3: Quantitative MOA-clustering strength")
print("=" * 60)

meta_known = meta[meta[MOA_COL].notna() & (meta[MOA_COL] != "unclear")].copy()
print(f"\nComparisons with known MOA: {len(meta_known):,} / {len(meta):,}")

# --- Overall agreement: Adjusted Rand Index ---
ari = adjusted_rand_score(meta_known["leiden_comp"], meta_known[MOA_COL])
print(f"\nAdjusted Rand Index (Leiden clusters vs MOA labels): {ari:.4f}")
print("  (0 = no better than random agreement, 1 = perfect agreement.")
print("   ARI is typically modest even for real signal, since Leiden")
print("   clusters and MOA categories are different partitions with")
print("   different granularities - a small positive value here is still")
print("   meaningful evidence of association, not proof of failure.)")

# --- Targeted enrichment test for the key cluster ---
print(f"\nHypergeometric enrichment test for cluster {KEY_CLUSTER}:")
N_total = len(meta_known)
cluster_mask = meta_known["leiden_comp"].astype(str) == str(KEY_CLUSTER)
n_cluster = cluster_mask.sum()
print(f"  Cluster {KEY_CLUSTER} size (known-MOA comparisons): {n_cluster:,} / {N_total:,}")

moa_in_cluster = meta_known.loc[cluster_mask, MOA_COL].value_counts()
top_moas_in_cluster = moa_in_cluster.head(6).index.tolist()

print(f"\n  {'MOA':<32} {'in_cluster':>10} {'total_in_data':>14} {'p-value':>12}")
for moa in top_moas_in_cluster:
    k = (meta_known.loc[cluster_mask, MOA_COL] == moa).sum()          # in cluster & this MOA
    K = (meta_known[MOA_COL] == moa).sum()                            # total this MOA
    n = n_cluster                                                     # cluster size
    N = N_total                                                       # total comparisons
    # P(X >= k) using hypergeometric survival function
    pval = hypergeom.sf(k - 1, N, K, n)
    print(f"  {moa:<32} {k:>10} {K:>14} {pval:>12.2e}")

print("\n  Interpretation: a small p-value (e.g. < 0.001) means this MOA is")
print("  significantly OVER-represented in cluster", KEY_CLUSTER,
      "beyond what random chance would predict given its overall frequency.")

# ═════════════════════════════════════════════
# CHECK 3: targeted higher-resolution rerun
# ═════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"CHECK 5: Higher-resolution rerun (resolution={RESOLUTION_HIGH})")
print(f"  Testing whether cluster {KEY_CLUSTER} stays intact or splits")
print("=" * 60)
t0 = time.time()

# Regenerate the PCs cheaply (same z-scored input as before) - reuses
# saved z-score params, does NOT redo t-SNE/NNMDS
parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "group_*.parquet")))
dfs = [pd.read_parquet(f) for f in parquet_files]
scores_df = pd.concat(dfs, axis=0)
del dfs
scores_df = scores_df[~scores_df.index.duplicated(keep='first')]

split_index = scores_df.index.str.split(r"\s*\|\s*", expand=True)
comp_metadata = pd.DataFrame({
    "drug":      split_index.get_level_values(0),
    "plate":     split_index.get_level_values(1),
    "cell_line": split_index.get_level_values(2),
}, index=scores_df.index)
mask = comp_metadata["plate"].astype(str) != EXCLUDE_PLATE
scores_df = scores_df[mask]
comp_metadata = comp_metadata[mask]
scores = np.nan_to_num(scores_df.values.astype(np.float32), nan=0.0)
del scores_df

gs_mean = np.load(os.path.join(OUTPUT_DIR, "geneset_zscore_mean.npy"))
gs_std  = np.load(os.path.join(OUTPUT_DIR, "geneset_zscore_std.npy"))
scores_z = (scores - gs_mean) / gs_std
del scores

if USE_GPU_PCA:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    scores_gpu = cp.array(scores_z)
    pca = cuPCA(n_components=N_PCS_LEIDEN)
    pcs = cp.asnumpy(pca.fit_transform(scores_gpu))
    del scores_gpu
else:
    from sklearn.decomposition import PCA as skPCA
    pca = skPCA(n_components=N_PCS_LEIDEN, random_state=RANDOM_STATE, svd_solver="randomized")
    pcs = pca.fit_transform(scores_z).astype(np.float32)
del scores_z

print(f"  PCA regenerated ({time.time()-t0:.0f}s)")

ad = anndata.AnnData(X=pcs)
ad.obs_names = comp_metadata.index
ad.obsm["X_pca_precomputed"] = pcs
sc.pp.neighbors(ad, use_rep="X_pca_precomputed", n_neighbors=N_NEIGHBORS)
sc.tl.leiden(ad, key_added="leiden_high_res", resolution=RESOLUTION_HIGH,
             random_state=RANDOM_STATE, flavor="leidenalg")
n_clusters_high = ad.obs["leiden_high_res"].nunique()
print(f"  Resolution {RESOLUTION_HIGH}: {n_clusters_high} clusters "
      f"(original resolution {LEIDEN_RESOLUTION}: {meta['leiden_comp'].nunique()} clusters)")

# Merge new cluster labels back onto original metadata for direct comparison
meta_aligned = meta.set_index(meta.index if meta.index.name else meta.iloc[:, 0].index) \
    if False else meta.copy()
meta_aligned = meta.copy()
meta_aligned["_key"] = meta_aligned["drug"].astype(str) + " | " + \
    meta_aligned["plate"].astype(str) + " | " + meta_aligned["cell_line"].astype(str)
new_labels = pd.Series(ad.obs["leiden_high_res"].values, index=ad.obs_names, name="leiden_high_res")
meta_aligned = meta_aligned.set_index(meta_aligned["_key"]).join(new_labels).reset_index(drop=True)

print(f"\n  How original cluster {KEY_CLUSTER} redistributes at higher resolution:")
key_cluster_mask = meta_aligned["leiden_comp"].astype(str) == str(KEY_CLUSTER)
redistribution = meta_aligned.loc[key_cluster_mask, "leiden_high_res"].value_counts()
print(redistribution)

if len(redistribution) == 1:
    print(f"\n  -> Cluster {KEY_CLUSTER} STAYS INTACT at resolution={RESOLUTION_HIGH} "
          "- a robust finding.")
else:
    print(f"\n  -> Cluster {KEY_CLUSTER} SPLITS into {len(redistribution)} sub-clusters "
          f"at resolution={RESOLUTION_HIGH}. Check the MOA composition of each "
          "sub-cluster below to see if this reveals finer mechanistic structure.")
    for sub_cluster in redistribution.index:
        sub_mask = meta_aligned["leiden_high_res"] == sub_cluster
        sub_known = meta_aligned[sub_mask & meta_aligned[MOA_COL].notna() &
                                  (meta_aligned[MOA_COL] != "unclear")]
        if len(sub_known) > 0:
            print(f"\n  Sub-cluster {sub_cluster} (n={sub_mask.sum()}) top MOAs:")
            print(f"    {sub_known[MOA_COL].value_counts().head(5).to_dict()}")

# Save full high-res MOA heatmap too, for completeness
meta_known_high = meta_aligned[meta_aligned[MOA_COL].notna() &
                                (meta_aligned[MOA_COL] != "unclear")]
if len(meta_known_high) > 0:
    ct = pd.crosstab(meta_known_high["leiden_high_res"], meta_known_high[MOA_COL])
    ct_norm = ct.div(ct.sum(axis=0), axis=1)
    g = sns.clustermap(
        ct_norm, cmap="YlOrRd", linewidths=0.1,
        figsize=(max(10, ct_norm.shape[1]*0.4), max(6, ct_norm.shape[0]*0.4)),
        cbar_kws={"label": "Fraction", "shrink": 0.7},
        xticklabels=True, yticklabels=True, method="ward", metric="euclidean",
        row_cluster=ct_norm.shape[0] > 1, col_cluster=ct_norm.shape[1] > 1,
    )
    g.ax_heatmap.set_title(f"Leiden vs MOA — resolution={RESOLUTION_HIGH}", fontsize=13, pad=10)
    g.ax_heatmap.set_xlabel("Mechanism of Action", fontsize=10)
    g.ax_heatmap.set_ylabel("Leiden Cluster", fontsize=10)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7, rotation=0)
    out_path = os.path.join(OUTPUT_DIR, f"comparison_leiden_vs_moa_res{RESOLUTION_HIGH}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out_path}")

meta_aligned.to_parquet(os.path.join(OUTPUT_DIR, "comparison_leiden_metadata_with_highres.parquet"))
print(f"\nSaved extended metadata -> "
      f"{OUTPUT_DIR}/comparison_leiden_metadata_with_highres.parquet")

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETE")
print("=" * 60)