"""
Drug-Level Leiden — Normalized Direction Variant + Higher-Resolution Rerun
==============================================================================
Builds on drug_level_leiden.py's saved outputs. Adds two things:

  [ADDITION 1] Magnitude-normalized clustering: each drug's z-scored
  gene-set vector is divided by its own L2 norm before PCA/Leiden. This
  forces every drug onto a unit sphere, so clustering reflects ONLY the
  shape/direction of its response profile, independent of how strongly it
  perturbs. 

  [ADDITION 2] One higher-resolution rerun testing whether the large, 
  MOA-mixed "positive branch" clusters from the original run (clusters 0 and 2,
  ~78 and ~75 drugs each, spanning many distinct MOAs) split into more mechanistically 
  coherent sub-groups at higher resolution.

Requires the same saved files as pc1_magnitude_check.py:
  drug_level_matrix_raw.parquet, drug_level_matrix_zscored.parquet,
  drug_leiden_metadata.parquet
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR = "/home/a/aangelopa/Thesis/Data/2ndtry/drug_level_leiden"

RAW_MATRIX_PATH     = f"{OUTPUT_DIR}/drug_level_matrix_raw.parquet"
ZSCORED_MATRIX_PATH = f"{OUTPUT_DIR}/drug_level_matrix_zscored.parquet"
META_PATH           = f"{OUTPUT_DIR}/drug_leiden_metadata.parquet"
MOA_COL             = "moa-fine"

N_PCS             = 30
N_NEIGHBORS       = 15
RANDOM_STATE      = 42

RESOLUTION_HIGH   = 1.0   # targeted rerun to test cluster-0/2 splitting

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD SAVED DATA
# ─────────────────────────────────────────────
print("Loading saved drug-level matrices and metadata...")
raw_matrix = pd.read_parquet(RAW_MATRIX_PATH)
z_matrix   = pd.read_parquet(ZSCORED_MATRIX_PATH)
meta       = pd.read_parquet(META_PATH)
meta       = meta.set_index("drug").loc[raw_matrix.index].reset_index()
print(f"  {z_matrix.shape[0]} drugs x {z_matrix.shape[1]} gene sets")


def run_leiden(X, resolution, key_added, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    """Shared PCA -> neighbors -> Leiden -> t-SNE helper."""
    ad = anndata.AnnData(X=X.astype(np.float32))
    ad.obs_names = z_matrix.index
    n_pcs_eff = min(n_pcs, ad.n_obs - 1, ad.n_vars)
    sc.tl.pca(ad, n_comps=n_pcs_eff, svd_solver="arpack", random_state=RANDOM_STATE)
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=min(n_neighbors, ad.n_obs - 1))
    sc.tl.leiden(ad, key_added=key_added, resolution=resolution,
                 random_state=RANDOM_STATE, flavor="leidenalg")
    sc.tl.tsne(ad, use_rep="X_pca", random_state=RANDOM_STATE)
    n_clusters = ad.obs[key_added].nunique()
    print(f"  [{key_added}] resolution={resolution} -> {n_clusters} clusters")
    return ad, n_clusters


def save_moa_heatmap(cluster_labels, moa_labels, title, out_path):
    df = pd.DataFrame({"cluster": cluster_labels, "moa": moa_labels})
    df = df[df["moa"].notna() & (df["moa"] != "unclear") & (df["moa"] != "unknown")]
    if len(df) == 0:
        print(f"    No known-MOA drugs available for {title} — skipping heatmap")
        return
    ct = pd.crosstab(df["cluster"], df["moa"])
    ct_norm = ct.div(ct.sum(axis=0), axis=1)
    g = sns.clustermap(
        ct_norm, cmap="YlOrRd", linewidths=0.1,
        figsize=(max(10, ct_norm.shape[1] * 0.4), max(6, ct_norm.shape[0] * 0.4)),
        cbar_kws={"label": "Fraction of drugs", "shrink": 0.7},
        xticklabels=True, yticklabels=True, method="ward", metric="euclidean",
        row_cluster=ct_norm.shape[0] > 1, col_cluster=ct_norm.shape[1] > 1,
    )
    g.ax_heatmap.set_title(title, fontsize=13, pad=10)
    g.ax_heatmap.set_xlabel("Mechanism of Action", fontsize=10)
    g.ax_heatmap.set_ylabel("Leiden Cluster", fontsize=10)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7, rotation=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out_path}")


moas = meta[MOA_COL].values

# ─────────────────────────────────────────────
# ADDITION 1: MAGNITUDE-NORMALIZED CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ADDITION 1: Magnitude-normalized (unit L2 norm) clustering")
print("=" * 60)

z_values = z_matrix.values
row_norms = np.sqrt((z_values ** 2).sum(axis=1, keepdims=True))
row_norms[row_norms == 0] = 1.0   # avoid div-by-zero for any all-zero row
z_normalized = z_values / row_norms

ad_norm, n_clusters_norm = run_leiden(
    z_normalized, resolution=0.5, key_added="leiden_normalized"
)

meta["leiden_normalized"] = ad_norm.obs["leiden_normalized"].values
meta["TSNE1_normalized"]  = ad_norm.obsm["X_tsne"][:, 0]
meta["TSNE2_normalized"]  = ad_norm.obsm["X_tsne"][:, 1]

save_moa_heatmap(
    meta["leiden_normalized"].values, moas,
    "Drug-level Leiden vs MOA — magnitude-normalized (direction only)",
    f"{OUTPUT_DIR}/drug_leiden_vs_moa_normalized.pdf"
)

# Quick comparison: does the branch (direction) split survive normalization?
print("\n  Branch composition within normalized clusters (sanity check):")
z_pc1_norm = ad_norm.obsm["X_pca"][:, 0]
branch_norm = np.where(z_pc1_norm > 0, "positive", "negative")
print(pd.crosstab(meta["leiden_normalized"], branch_norm))

# ─────────────────────────────────────────────
# ADDITION 2: TARGETED HIGHER-RESOLUTION RERUN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"ADDITION 2: Higher-resolution rerun (resolution={RESOLUTION_HIGH})")
print("  Testing whether broad positive-branch clusters split further")
print("=" * 60)

ad_high, n_clusters_high = run_leiden(
    z_values, resolution=RESOLUTION_HIGH, key_added="leiden_high_res"
)
meta["leiden_high_res"] = ad_high.obs["leiden_high_res"].values
meta["TSNE1_high_res"]  = ad_high.obsm["X_tsne"][:, 0]
meta["TSNE2_high_res"]  = ad_high.obsm["X_tsne"][:, 1]

save_moa_heatmap(
    meta["leiden_high_res"].values, moas,
    f"Drug-level Leiden vs MOA — resolution={RESOLUTION_HIGH}",
    f"{OUTPUT_DIR}/drug_leiden_vs_moa_res{RESOLUTION_HIGH}.pdf"
)

# Compare: how did the original clusters 0 and 2 (broad positive-branch,
# per your earlier crosstab) get redistributed at higher resolution?
if "leiden_drug" in meta.columns:
    print("\n  How original clusters map onto higher-resolution clusters:")
    print(pd.crosstab(meta["leiden_drug"], meta["leiden_high_res"]))

# ─────────────────────────────────────────────
# SAVE UPDATED METADATA
# ─────────────────────────────────────────────
meta.to_parquet(f"{OUTPUT_DIR}/drug_leiden_metadata_extended.parquet")
print(f"\nSaved extended metadata -> {OUTPUT_DIR}/drug_leiden_metadata_extended.parquet")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Original clusters (res=0.5, unnormalized): (see drug_leiden_metadata.parquet)")
print(f"  Normalized clusters (res=0.5, direction-only): {n_clusters_norm}")
print(f"  Higher-res clusters (res={RESOLUTION_HIGH}, unnormalized):   {n_clusters_high}")
print("\n  Compare drug_leiden_vs_moa_normalized.pdf against your original")
print("  drug_leiden_vs_moa.pdf: if MOA groupings hold up or sharpen, that's")
print("  evidence of mechanism-driven (not just magnitude-driven) clustering.")