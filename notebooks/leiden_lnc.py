"""
Leiden Clustering on saved lncRNA PCA
=====================================
Uses the PCA and metadata already saved by exploratory_lncrna.py.
No model loading, no GPU needed — runs on login node.

Outputs:
  leiden_vs_cellline.pdf
  leiden_vs_plate.pdf
  leiden_vs_cellcycle.pdf
  leiden_vs_drug_top30.pdf
  umap_leiden_cellline.pdf
  umap_leiden_plate_cycle.pdf
  lncrna_leiden_metadata.parquet
"""

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR = Path.home() / "Thesis"
INPUT_DIR = BASE_DIR / "Data" / "exploratory_lncrna"
OUTPUT_DIR = BASE_DIR / "Results" / "leiden_lncrna"

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL = "plate"
PHASE_COL = "phase"
DRUG_COL = "drug"

LEIDEN_RESOLUTION = 0.5
N_NEIGHBORS = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD SAVED PCA AND METADATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading saved PCA and metadata")
print("=" * 60)
t0 = time.time()

pcs = np.load(INPUT_DIR / "lncrna_pca_50.npy")
meta = pd.read_parquet(INPUT_DIR / "lncrna_exploratory_metadata.parquet")

print(f"  PCA shape: {pcs.shape}")
print(f"  Metadata: {meta.shape}")
print(f"  Cell lines: {meta[CELL_LINE_COL].nunique()}")
print(f"  Plates: {meta[PLATE_COL].nunique()}")
print(f"  Drugs: {meta[DRUG_COL].nunique()}")

n_pcs = pcs.shape[1]

# ─────────────────────────────────────────────
# 2. BUILD ANNDATA FOR SCANPY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building AnnData")
print("=" * 60)

adata_cluster = anndata.AnnData(obs=meta.reset_index(drop=True))
adata_cluster.obsm["X_pca_lncrna"] = pcs

print(f"  AnnData: {adata_cluster.n_obs} cells")

# ─────────────────────────────────────────────
# 3. NEIGHBORS + LEIDEN + UMAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Neighbors → Leiden → UMAP")
print("=" * 60)
t1 = time.time()

sc.pp.neighbors(adata_cluster, use_rep="X_pca_lncrna",
                n_neighbors=N_NEIGHBORS, n_pcs=n_pcs)

LEIDEN_KEY = "leiden_lncrna"
sc.tl.leiden(adata_cluster, key_added=LEIDEN_KEY, resolution=LEIDEN_RESOLUTION)
n_clusters = adata_cluster.obs[LEIDEN_KEY].nunique()
print(f"  Leiden clusters: {n_clusters} (resolution={LEIDEN_RESOLUTION})")

sc.tl.umap(adata_cluster, min_dist=0.3)
print(f"  Took {time.time() - t1:.0f}s")

# Save metadata with clusters
meta_out = adata_cluster.obs.copy()
meta_out["UMAP1"] = adata_cluster.obsm["X_umap"][:, 0]
meta_out["UMAP2"] = adata_cluster.obsm["X_umap"][:, 1]
meta_out.to_parquet(OUTPUT_DIR / "lncrna_leiden_metadata.parquet")

umap_coords = adata_cluster.obsm["X_umap"]

# ─────────────────────────────────────────────
# 4. CONFUSION MATRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Confusion matrices")
print("=" * 60)

# --- Leiden vs Cell Line ---
ct_cl = pd.crosstab(meta_out[LEIDEN_KEY], meta_out[CELL_LINE_COL])
norm_ct_cl = ct_cl.div(ct_cl.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(20, max(8, n_clusters * 0.35)))
sns.heatmap(norm_ct_cl, ax=ax, cmap="YlOrRd", linewidths=0.1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Fraction of cells", "shrink": 0.7})
ax.set_title(f"Leiden clusters vs Cell Lines (lncRNA, res={LEIDEN_RESOLUTION})", fontsize=13)
ax.set_xlabel("Cell Line", fontsize=10)
ax.set_ylabel("Leiden Cluster", fontsize=10)
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "leiden_vs_cellline.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → leiden_vs_cellline.pdf")

# --- Leiden vs Plate ---
ct_plate = pd.crosstab(meta_out[LEIDEN_KEY], meta_out[PLATE_COL])
norm_ct_plate = ct_plate.div(ct_plate.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, max(8, n_clusters * 0.35)))
sns.heatmap(norm_ct_plate, ax=ax, cmap="YlOrRd", linewidths=0.1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Fraction of cells", "shrink": 0.7})
ax.set_title(f"Leiden clusters vs Plates (lncRNA, res={LEIDEN_RESOLUTION})", fontsize=13)
ax.set_xlabel("Plate", fontsize=10)
ax.set_ylabel("Leiden Cluster", fontsize=10)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "leiden_vs_plate.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → leiden_vs_plate.pdf")

# --- Leiden vs Cell Cycle ---
ct_phase = pd.crosstab(meta_out[LEIDEN_KEY], meta_out[PHASE_COL])
norm_ct_phase = ct_phase.div(ct_phase.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(6, max(8, n_clusters * 0.35)))
sns.heatmap(norm_ct_phase, ax=ax, cmap="YlOrRd", linewidths=0.1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Fraction of cells", "shrink": 0.7})
ax.set_title(f"Leiden clusters vs Cell Cycle (lncRNA, res={LEIDEN_RESOLUTION})", fontsize=13)
ax.set_xlabel("Phase", fontsize=10)
ax.set_ylabel("Leiden Cluster", fontsize=10)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "leiden_vs_cellcycle.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → leiden_vs_cellcycle.pdf")

# --- Leiden vs Drug (top 30) ---
drug_counts = meta_out[meta_out[DRUG_COL] != "DMSO_TF"][DRUG_COL].value_counts()
top_drugs = drug_counts.head(30).index.tolist()
meta_drugs = meta_out[meta_out[DRUG_COL].isin(top_drugs)]

ct_drug = pd.crosstab(meta_drugs[LEIDEN_KEY], meta_drugs[DRUG_COL])
norm_ct_drug = ct_drug.div(ct_drug.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(16, max(8, n_clusters * 0.35)))
sns.heatmap(norm_ct_drug, ax=ax, cmap="YlOrRd", linewidths=0.1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Fraction of cells", "shrink": 0.7})
ax.set_title(f"Leiden clusters vs Top 30 Drugs (lncRNA, res={LEIDEN_RESOLUTION})", fontsize=13)
ax.set_xlabel("Drug", fontsize=10)
ax.set_ylabel("Leiden Cluster", fontsize=10)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "leiden_vs_drug_top30.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → leiden_vs_drug_top30.pdf")

# --- Leiden vs MOA ---
DRUG_META_PATH = BASE_DIR / "Data" / "drug_metadata.parquet"
drug_meta = pd.read_parquet(DRUG_META_PATH)
drug_meta["drug"] = drug_meta["drug"].str.strip()
MOA_COL = "moa-fine"

# Merge MOA into metadata
meta_moa = meta_out.copy()
meta_moa[DRUG_COL] = meta_moa[DRUG_COL].str.strip()
meta_moa = meta_moa.merge(drug_meta[["drug", MOA_COL]], left_on=DRUG_COL, right_on="drug",
                          how="left", suffixes=("", "_meta"))

# Filter to known MOA only (exclude unclear and NaN)
meta_moa_known = meta_moa[(meta_moa[MOA_COL].notna()) & (meta_moa[MOA_COL] != "unclear")]
print(f"  Cells with known MOA: {len(meta_moa_known):,} / {len(meta_moa):,}")

ct_moa = pd.crosstab(meta_moa_known[LEIDEN_KEY], meta_moa_known[MOA_COL])
norm_ct_moa = ct_moa.div(ct_moa.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(16, max(8, n_clusters * 0.35)))
sns.heatmap(norm_ct_moa, ax=ax, cmap="YlOrRd", linewidths=0.1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Fraction of cells", "shrink": 0.7})
ax.set_title(f"Leiden clusters vs MOA (lncRNA, res={LEIDEN_RESOLUTION})", fontsize=13)
ax.set_xlabel("Mechanism of Action", fontsize=10)
ax.set_ylabel("Leiden Cluster", fontsize=10)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "leiden_vs_moa.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → leiden_vs_moa.pdf")

# ─────────────────────────────────────────────
# 5. UMAP PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: UMAP plots")
print("=" * 60)

scatter_kw = dict(s=0.5, alpha=0.3, rasterized=True, edgecolors="none")

# Color maps
all_colors = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])

# Cluster colors
clusters = meta_out[LEIDEN_KEY].values
unique_clusters = sorted(set(clusters), key=int)
cluster_cmap = plt.cm.tab20(np.linspace(0, 1, 20))
cluster_colors = {c: cluster_cmap[int(c) % 20] for c in unique_clusters}

# Cell line colors
cell_lines = meta_out[CELL_LINE_COL].values
unique_cls = sorted(set(cell_lines))
cl_colors = {cl: all_colors[i % len(all_colors)] for i, cl in enumerate(unique_cls)}

# === Leiden + Cell Line side by side ===
fig, axes = plt.subplots(1, 2, figsize=(24, 9))

axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=[cluster_colors[c] for c in clusters], **scatter_kw)
axes[0].set_title(f"lncRNA UMAP — Leiden Clusters (n={n_clusters})", fontsize=13)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
if n_clusters <= 50:
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cluster_colors[c], markersize=6, label=c)
               for c in unique_clusters]
    axes[0].legend(handles=handles, title="Cluster",
                   bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=6, markerscale=1.5, ncol=2, title_fontsize=8)

axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=[cl_colors[cl] for cl in cell_lines], **scatter_kw)
axes[1].set_title("lncRNA UMAP — Cell Line", fontsize=13)
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
handles_cl = [plt.Line2D([0], [0], marker='o', color='w',
              markerfacecolor=cl_colors[cl], markersize=5, label=cl)
              for cl in unique_cls]
axes[1].legend(handles=handles_cl, title="Cell line",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=4, markerscale=1.5, ncol=3, title_fontsize=7)

plt.suptitle(f"Leiden clustering on lncRNA expression ({n_pcs} PCs, res={LEIDEN_RESOLUTION})",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_leiden_cellline.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → umap_leiden_cellline.pdf")

# === Clusters + Plate + Cell Cycle (3 panels) ===
fig, axes = plt.subplots(1, 3, figsize=(28, 8))

axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=[cluster_colors[c] for c in clusters], **scatter_kw)
axes[0].set_title(f"Leiden Clusters (n={n_clusters})", fontsize=13)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

plates = meta_out[PLATE_COL].values
unique_plates = sorted(set(plates))
plate_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_plates), 12)))
plate_colors = {p: plate_cmap[i % len(plate_cmap)] for i, p in enumerate(unique_plates)}

axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=[plate_colors[p] for p in plates], **scatter_kw)
axes[1].set_title("Plate (Batch)", fontsize=13)
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
handles_plate = [plt.Line2D([0], [0], marker='o', color='w',
                 markerfacecolor=plate_colors[p], markersize=6, label=p)
                 for p in unique_plates]
axes[1].legend(handles=handles_plate, title="Plate",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7, markerscale=1.2, title_fontsize=8)

phases = meta_out[PHASE_COL].values
phase_colors = {"G1": "#1f77b4", "S": "#ff7f0e", "G2M": "#2ca02c"}
axes[2].scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=[phase_colors.get(p, "gray") for p in phases], **scatter_kw)
axes[2].set_title("Cell Cycle Phase", fontsize=13)
axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")
handles_phase = [plt.Line2D([0], [0], marker='o', color='w',
                 markerfacecolor=phase_colors[p], markersize=8, label=p)
                 for p in ["G1", "S", "G2M"]]
axes[2].legend(handles=handles_phase, title="Phase",
               loc="upper right", fontsize=9, markerscale=1.5, title_fontsize=10)

plt.suptitle("lncRNA UMAP — Factors driving clustering", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_leiden_plate_cycle.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → umap_leiden_plate_cycle.pdf")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'=' * 60}")
print("DONE")
print(f"  Cells:          {pcs.shape[0]:,}")
print(f"  PCs:            {n_pcs}")
print(f"  Leiden clusters: {n_clusters}")
print(f"  Resolution:     {LEIDEN_RESOLUTION}")
print(f"  Total time:     {total:.0f}s")
print(f"\nOutputs in {OUTPUT_DIR}:")
print(f"  leiden_vs_cellline.pdf")
print(f"  leiden_vs_plate.pdf")
print(f"  leiden_vs_cellcycle.pdf")
print(f"  leiden_vs_drug_top30.pdf")
print(f"  umap_leiden_cellline.pdf")
print(f"  umap_leiden_plate_cycle.pdf")
print(f"  lncrna_leiden_metadata.parquet")
print(f"{'=' * 60}")