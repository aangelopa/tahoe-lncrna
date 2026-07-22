"""
Drug-Level Leiden Clustering on Differential Vision Scores
=============================================================
 It clusters DRUGS directly, using the differential Vision score matrix
already validated against E-distance.


Input:  Directory of parquet files (group_XXXX.parquet), same as your
        dim_reduction_v2.py pipeline
        rows = comparisons: 'drug | plate | cell_line'
        cols = gene sets (differential Vision scores)

Output: drug_level_matrix.parquet   (n_drugs x n_genesets, z-scored)
        drug_leiden_metadata.parquet (drug, leiden cluster, MOA, n_comparisons)
        drug_tsne_embedding.npy
        drug_leiden_vs_moa.pdf
        drug_embeddings_overview.pdf
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import glob
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = "/home/a/aangelopa/Thesis/Data/diff_vision_chunks"
OUTPUT_DIR     = "/home/a/aangelopa/Thesis/Data/2ndtry/drug_level_leiden"
DRUG_META_PATH = "/home/a/aangelopa/Thesis/Data/drug_metadata.parquet"

MOA_COL       = "moa-fine"
EXCLUDE_PLATE = "14"
MIN_COMPARISONS_PER_DRUG = 3   # drop drugs tested in fewer comparisons (noisy median)

N_PCS             = 30   # fewer PCs than the comparison-level pipeline —
                          # you now have ~379 drugs, not ~63,000 comparisons,
                          # so fewer components avoid overfitting noise
LEIDEN_RESOLUTION = 0.5
N_NEIGHBORS       = 15
RANDOM_STATE      = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. MERGE ALL PARQUET FILES  (same as dim_reduction_v2.py)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging parquet files")
print("=" * 60)
t0 = time.time()

parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "group_*.parquet")))
print(f"  Found {len(parquet_files)} parquet files")
if len(parquet_files) == 0:
    raise FileNotFoundError(f"No group_*.parquet files found in {DATA_DIR}")

dfs = [pd.read_parquet(f) for f in parquet_files]
scores_df = pd.concat(dfs, axis=0)
del dfs

n_dups = scores_df.index.duplicated().sum()
if n_dups > 0:
    print(f"  WARNING: {n_dups} duplicate indices — dropping duplicates")
    scores_df = scores_df[~scores_df.index.duplicated(keep='first')]

print(f"  Merged shape: {scores_df.shape}")

# ─────────────────────────────────────────────
# 2. PARSE METADATA + EXCLUDE PLATE 14
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Parsing metadata, excluding replication plate")
print("=" * 60)

split_index = scores_df.index.str.split(r"\s*\|\s*", expand=True)
metadata = pd.DataFrame({
    "drug":      split_index.get_level_values(0),
    "plate":     split_index.get_level_values(1),
    "cell_line": split_index.get_level_values(2),
}, index=scores_df.index)

mask = metadata["plate"].astype(str) != EXCLUDE_PLATE
scores_df = scores_df[mask]
metadata  = metadata[mask]
print(f"  Comparisons after excluding plate {EXCLUDE_PLATE}: {len(metadata):,}")
print(f"  Unique drugs (comparison-level): {metadata['drug'].nunique()}")

gene_set_names = list(scores_df.columns)
scores = np.nan_to_num(scores_df.values.astype(np.float32), nan=0.0)
del scores_df

# ─────────────────────────────────────────────
# 3. AGGREGATE TO DRUG LEVEL  (THE KEY STEP — nullifies cell_line/plate)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Aggregating to drug-level median (nullifying cell_line/plate)")
print("=" * 60)

df_full = pd.DataFrame(scores, index=metadata.index, columns=gene_set_names)
df_full["drug"] = metadata["drug"].values

# Count comparisons per drug before aggregating, for filtering + reporting
n_comparisons_per_drug = df_full.groupby("drug").size()
print(f"  Drugs with >= {MIN_COMPARISONS_PER_DRUG} comparisons: "
      f"{(n_comparisons_per_drug >= MIN_COMPARISONS_PER_DRUG).sum()} "
      f"/ {len(n_comparisons_per_drug)}")

# THE AGGREGATION: median differential score per gene set, per drug.
# After this line, cell_line and plate no longer exist in the data at all —
# there is exactly one row per drug.
drug_matrix = df_full.groupby("drug")[gene_set_names].median()

valid_drugs = n_comparisons_per_drug[n_comparisons_per_drug >= MIN_COMPARISONS_PER_DRUG].index
drug_matrix = drug_matrix.loc[valid_drugs]
n_comparisons_per_drug = n_comparisons_per_drug.loc[valid_drugs]

print(f"  Drug-level matrix: {drug_matrix.shape[0]} drugs x {drug_matrix.shape[1]} gene sets")
del df_full, scores

# ─────────────────────────────────────────────
# 4. Z-SCORE PER GENE SET  (recommended fix discussed above)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Z-scoring per gene set before PCA")
print("=" * 60)
print("  Gene sets vary in size (15-500 genes) and raw variance;")
print("  z-scoring prevents high-variance sets from dominating PCA.")

gs_mean = drug_matrix.mean(axis=0)
gs_std  = drug_matrix.std(axis=0) + 1e-8
drug_matrix_z = (drug_matrix - gs_mean) / gs_std

drug_matrix.to_parquet(os.path.join(OUTPUT_DIR, "drug_level_matrix_raw.parquet"))
drug_matrix_z.to_parquet(os.path.join(OUTPUT_DIR, "drug_level_matrix_zscored.parquet"))

# ─────────────────────────────────────────────
# 5. PCA + NEIGHBORS + LEIDEN  (scanpy, CPU — matrix is small: ~379 x n_genesets)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: PCA, neighbors, Leiden clustering")
print("=" * 60)

adata = anndata.AnnData(X=drug_matrix_z.values.astype(np.float32))
adata.obs_names = drug_matrix_z.index
adata.obs["n_comparisons"] = n_comparisons_per_drug.values

n_pcs_eff = min(N_PCS, adata.n_obs - 1, adata.n_vars)
sc.tl.pca(adata, n_comps=n_pcs_eff, svd_solver="arpack", random_state=RANDOM_STATE)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=min(N_NEIGHBORS, adata.n_obs - 1))
sc.tl.leiden(adata, key_added="leiden_drug", resolution=LEIDEN_RESOLUTION,
             random_state=RANDOM_STATE, flavor="leidenalg")
sc.tl.tsne(adata, use_rep="X_pca", random_state=RANDOM_STATE)

n_clusters = adata.obs["leiden_drug"].nunique()
print(f"  Drugs: {adata.n_obs}  |  Leiden clusters: {n_clusters}")
print(f"  (Compare n_clusters to n_drugs={adata.n_obs} — clusters should be")
print(f"   far fewer than drugs, and CANNOT equal n_cell_lines since cell_line")
print(f"   is not present in this matrix at all.)")

# ─────────────────────────────────────────────
# 6. MERGE MOA + SAVE METADATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Merging MOA annotations")
print("=" * 60)

meta_out = pd.DataFrame({
    "drug":          adata.obs_names,
    "leiden_drug":   adata.obs["leiden_drug"].values,
    "n_comparisons": adata.obs["n_comparisons"].values,
    "TSNE1":         adata.obsm["X_tsne"][:, 0],
    "TSNE2":         adata.obsm["X_tsne"][:, 1],
})

if os.path.exists(DRUG_META_PATH):
    drug_meta = pd.read_parquet(DRUG_META_PATH)
    drug_meta["drug"] = drug_meta["drug"].str.strip()
    meta_out["drug_stripped"] = meta_out["drug"].str.strip()
    meta_out = meta_out.merge(
        drug_meta[["drug", MOA_COL]],
        left_on="drug_stripped", right_on="drug",
        how="left", suffixes=("", "_meta")
    )
    n_known_moa = meta_out[MOA_COL].notna().sum()
    print(f"  Drugs with known MOA: {n_known_moa} / {len(meta_out)}")
else:
    print(f"  WARNING: {DRUG_META_PATH} not found — skipping MOA merge")
    meta_out[MOA_COL] = np.nan

meta_out.to_parquet(os.path.join(OUTPUT_DIR, "drug_leiden_metadata.parquet"))
np.save(os.path.join(OUTPUT_DIR, "drug_tsne_embedding.npy"), adata.obsm["X_tsne"])

# ─────────────────────────────────────────────
# 7. LEIDEN VS MOA HEATMAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Leiden vs MOA heatmap")
print("=" * 60)

meta_known = meta_out[meta_out[MOA_COL].notna() & (meta_out[MOA_COL] != "unclear")]
print(f"  Drugs with usable MOA for heatmap: {len(meta_known)}")

if len(meta_known) > 0:
    ct = pd.crosstab(meta_known["leiden_drug"], meta_known[MOA_COL])
    ct_norm = ct.div(ct.sum(axis=0), axis=1)

    g = sns.clustermap(
        ct_norm, cmap="YlOrRd", linewidths=0.1,
        figsize=(max(10, ct_norm.shape[1] * 0.4), max(6, ct_norm.shape[0] * 0.4)),
        cbar_kws={"label": "Fraction of drugs", "shrink": 0.7},
        xticklabels=True, yticklabels=True,
        method="ward", metric="euclidean",
        row_cluster=ct_norm.shape[0] > 1, col_cluster=ct_norm.shape[1] > 1,
    )
    g.ax_heatmap.set_title("Drug-level Leiden vs MOA (cell_line/plate nullified)",
                           fontsize=13, pad=10)
    g.ax_heatmap.set_xlabel("Mechanism of Action", fontsize=10)
    g.ax_heatmap.set_ylabel("Leiden Cluster (drug-level)", fontsize=10)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7, rotation=0)
    out_heatmap = os.path.join(OUTPUT_DIR, "drug_leiden_vs_moa.pdf")
    plt.savefig(out_heatmap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_heatmap}")

# ─────────────────────────────────────────────
# 8. VISUALIZATION: t-SNE colored by Leiden cluster + MOA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Plotting drug-level t-SNE")
print("=" * 60)

def make_color_map(values):
    unique_vals = sorted(set(values))
    all_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_vals)}
    return unique_vals, color_map

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

clusters = meta_out["leiden_drug"].values
unique_clusters, cluster_color = make_color_map(clusters)
axes[0].scatter(meta_out["TSNE1"], meta_out["TSNE2"],
                c=[cluster_color[c] for c in clusters], s=40, alpha=0.8)
for _, row in meta_out.iterrows():
    axes[0].annotate(row["drug"][:12], (row["TSNE1"], row["TSNE2"]), fontsize=4, alpha=0.6)
axes[0].set_title(f"Drug-level t-SNE - Leiden (n={n_clusters})", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

moas = meta_out[MOA_COL].fillna("unknown").values
unique_moas, moa_color = make_color_map(moas)
axes[1].scatter(meta_out["TSNE1"], meta_out["TSNE2"],
                c=[moa_color[m] for m in moas], s=40, alpha=0.8)
axes[1].set_title("Drug-level t-SNE - MOA", fontsize=13)
axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
if len(unique_moas) <= 40:
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=moa_color[m], markersize=6, label=m)
               for m in unique_moas]
    axes[1].legend(handles=handles, title="MOA", bbox_to_anchor=(1.02, 1),
                   loc="upper left", fontsize=5, markerscale=1.2,
                   ncol=max(1, len(unique_moas) // 25))

plt.tight_layout()
out_overview = os.path.join(OUTPUT_DIR, "drug_embeddings_overview.pdf")
plt.savefig(out_overview, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {out_overview}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print(f"  Drugs clustered      : {adata.n_obs}")
print(f"  Gene sets            : {len(gene_set_names)}")
print(f"  Leiden clusters       : {n_clusters}")
print(f"  Total time            : {total:.0f}s ({total/60:.1f} min)")
print(f"  Outputs in            : {OUTPUT_DIR}")
print("=" * 60)