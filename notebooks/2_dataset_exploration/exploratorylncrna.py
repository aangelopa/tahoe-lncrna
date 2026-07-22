"""
Exploratory Analysis with lncRNA genes

Outputs:
  1. Confusion table: plates × cell lines (coverage)
  2. Drug × cell line heatmap (drug-cell line distribution)

  Not included in the Methods and Results report:
  3. UMAP from PCA on lncRNA SCVI-normalized expression
  4. UMAP colored by: cell line, plate, cell cycle, drug
  5. Saves PCA output for future reuse

All using only lncRNA genes from lncrna_sets.txt
"""

import numpy as np
import pandas as pd
import torch
import anndata
import scvi.hub
import cupy as cp
from cuml.manifold import UMAP as cuUMAP
from cuml.decomposition import PCA as cuPCA
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
DATA_DIR = BASE_DIR / "Data"
CACHE_DIR = Path("/home/aangelopa/Thesis/Data/tahoe_cache")
SNAPSHOT_PATH = Path(
    "/home/aangelopa/Thesis/Data/tahoe_cache/"
    "models--vevotx--Tahoe-100M-SCVI-v1/snapshots/"
    "b5283a73fbbed812a95264ace360da538b20af89"
)
DATA_PATH = SNAPSHOT_PATH / "adata.h5ad"
GENE_SET_FILE = DATA_DIR / "lncrna_sets.txt"
OUTPUT_DIR = BASE_DIR / "Data" / "exploratory_lncrna"

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL = "plate"
PHASE_COL = "phase"
DRUG_COL = "drug"

N_CELLS_TOTAL = 200_000
N_TOP_CELL_LINES = 47
N_PCS = 50
LIBRARY_SIZE = 10e4
BATCH_SIZE = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD lncRNA GENES
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading lncRNA gene list")
print("=" * 60)

lnc_genes = set()
n_sets = 0
with open(GENE_SET_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            genes = parts[2].split(",")
            lnc_genes.update(g.strip() for g in genes if g.strip())
            n_sets += 1

lnc_genes = sorted(lnc_genes)
print(f"  Gene sets: {n_sets}")
print(f"  Unique lncRNA genes: {len(lnc_genes)}")

# ─────────────────────────────────────────────
# 2. LOAD MODEL AND DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Loading model and data")
print("=" * 60)
t0 = time.time()

tahoe_hubmodel = scvi.hub.HubModel.pull_from_huggingface_hub(
    repo_name="vevotx/Tahoe-100M-SCVI-v1",
    cache_dir=CACHE_DIR,
)
tahoe = tahoe_hubmodel.model
tahoe.to_device(DEVICE)
print(f"  Model on: {DEVICE}")

adata = anndata.read_h5ad(DATA_PATH)
print(f"  Full adata: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Filter valid lncRNA genes
valid_genes = [g for g in lnc_genes if g in adata.var_names]
print(f"  Valid lncRNA genes in model: {len(valid_genes)} / {len(lnc_genes)}")

print(f"  Loading took {time.time() - t0:.0f}s")

# ─────────────────────────────────────────────
# 3. CONFUSION TABLE: PLATES × CELL LINES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Plates × Cell Lines confusion table")
print("=" * 60)

# Use full adata for this (no subsampling)
ct = pd.crosstab(adata.obs[PLATE_COL], adata.obs[CELL_LINE_COL])
print(f"  Shape: {ct.shape} (plates × cell lines)")

fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(
    ct, ax=ax, cmap="YlOrRd", linewidths=0.1, linecolor="gray",
    xticklabels=True, yticklabels=True,
    cbar_kws={"label": "Number of cells"}
)
ax.set_title("Cell coverage: Plates × Cell Lines", fontsize=14)
ax.set_xlabel("Cell Line")
ax.set_ylabel("Plate")
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_plate_cellline.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → confusion_plate_cellline.pdf")

# Save as CSV too
ct.to_csv(OUTPUT_DIR / "confusion_plate_cellline.csv")

# ─────────────────────────────────────────────
# 4. DRUG × CELL LINE HEATMAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Drug × Cell Line distribution heatmap")
print("=" * 60)

# Use a subsample for this (full data is too large for groupby)
np.random.seed(42)
sub_idx = np.random.choice(adata.n_obs, min(1_000_000, adata.n_obs), replace=False)
sub_obs = adata.obs.iloc[sub_idx]

drug_cl = pd.crosstab(sub_obs[DRUG_COL], sub_obs[CELL_LINE_COL])
# Normalize by column (per cell line)
norm_drug_cl = drug_cl.div(drug_cl.sum(axis=0), axis=1)

print(f"  Shape: {drug_cl.shape} (drugs × cell lines)")
print(f"  Unique drugs: {drug_cl.shape[0]}")

fig, ax = plt.subplots(figsize=(16, 22))
sns.heatmap(
    norm_drug_cl, ax=ax, cmap="viridis", linewidths=0.05,
    xticklabels=True, yticklabels=True,
    cbar_kws={"label": "Fraction of cells (per cell line)", "shrink": 0.5}
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=3)
ax.set_xlabel("Cell Line", fontsize=10)
ax.set_ylabel("Drug", fontsize=10)
ax.set_title("Drug × Cell Line distribution (normalized per cell line)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_drug_cellline.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → heatmap_drug_cellline.pdf")

# ─────────────────────────────────────────────
# 5. SUBSAMPLE FOR EXPRESSION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Subsampling cells for lncRNA expression analysis")
print("=" * 60)

cl_counts = adata.obs[CELL_LINE_COL].value_counts()
top_cls = cl_counts.head(N_TOP_CELL_LINES).index.tolist()
cells_per_cl = N_CELLS_TOTAL // N_TOP_CELL_LINES
print(f"  Top {N_TOP_CELL_LINES} cell lines, {cells_per_cl} cells each")

np.random.seed(42)
sampled_indices = []
for cl in top_cls:
    cl_indices = np.where(adata.obs[CELL_LINE_COL] == cl)[0]
    n_available = len(cl_indices)
    if n_available >= cells_per_cl:
        chosen = np.random.choice(cl_indices, cells_per_cl, replace=False)
    else:
        chosen = cl_indices
        print(f"  WARNING: {cl} has only {n_available} cells")
    sampled_indices.append(chosen)

sampled_indices = np.concatenate(sampled_indices)
np.random.shuffle(sampled_indices)
subset = adata[sampled_indices].copy()
print(f"  Subset: {subset.n_obs:,} cells")

# ─────────────────────────────────────────────
# 6. GET NORMALIZED lncRNA EXPRESSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SCVI-normalized lncRNA expression")
print("=" * 60)
t1 = time.time()

print(f"  Generating expression for {len(valid_genes)} genes × {subset.n_obs:,} cells...")
expr = tahoe.get_normalized_expression(
    subset,
    gene_list=valid_genes,
    library_size=LIBRARY_SIZE,
    batch_size=BATCH_SIZE,
)

if isinstance(expr, pd.DataFrame):
    expr_matrix = expr.values.astype(np.float32)
else:
    expr_matrix = np.array(expr, dtype=np.float32)

print(f"  Expression matrix: {expr_matrix.shape}")
print(f"  Took {time.time() - t1:.0f}s")

# ─────────────────────────────────────────────
# 7. PCA (save for reuse)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: PCA on lncRNA expression")
print("=" * 60)
t2 = time.time()

n_pcs = min(N_PCS, expr_matrix.shape[1])
print(f"  PCA: {expr_matrix.shape[1]} genes → {n_pcs} PCs")

expr_gpu = cp.array(expr_matrix)
pca = cuPCA(n_components=n_pcs)
pcs = cp.asnumpy(pca.fit_transform(expr_gpu))
del expr_gpu, expr_matrix

ev = cp.asnumpy(pca.explained_variance_ratio_).cumsum()
print(f"  Variance explained by {n_pcs} PCs: {ev[-1]*100:.1f}%")

# Save PCA for future reuse
np.save(OUTPUT_DIR / "lncrna_pca_50.npy", pcs)
np.save(OUTPUT_DIR / "lncrna_pca_variance.npy", ev)
print(f"  Saved → lncrna_pca_50.npy")
print(f"  PCA took {time.time() - t2:.0f}s")

# ─────────────────────────────────────────────
# 8. UMAP (cuML GPU)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: UMAP on lncRNA PCA")
print("=" * 60)
t3 = time.time()

print(f"  UMAP on {pcs.shape[0]:,} cells × {pcs.shape[1]} PCs...")
umap = cuUMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
umap_embedding = cp.asnumpy(umap.fit_transform(cp.array(pcs.astype(np.float32))))

print(f"  UMAP shape: {umap_embedding.shape}")
print(f"  UMAP took {time.time() - t3:.0f}s")

# Save
np.save(OUTPUT_DIR / "lncrna_umap_embedding.npy", umap_embedding)

# Save metadata
meta = subset.obs[[CELL_LINE_COL, PLATE_COL, PHASE_COL, DRUG_COL]].copy()
meta.to_parquet(OUTPUT_DIR / "lncrna_exploratory_metadata.parquet")

# ─────────────────────────────────────────────
# 9. UMAP PLOTS (4 panels)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: UMAP plots")
print("=" * 60)

scatter_kw = dict(s=0.5, alpha=0.3, rasterized=True, edgecolors="none")

# --- Color maps ---
all_colors = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])

# Cell line colors
cell_lines = meta[CELL_LINE_COL].values
unique_cls = sorted(set(cell_lines))
cl_colors = {cl: all_colors[i % len(all_colors)] for i, cl in enumerate(unique_cls)}

# Plate colors
plates = meta[PLATE_COL].values
unique_plates = sorted(set(plates))
plate_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_plates), 12)))
plate_colors = {p: plate_cmap[i % len(plate_cmap)] for i, p in enumerate(unique_plates)}

# Phase colors
phase_colors = {"G1": "#1f77b4", "S": "#ff7f0e", "G2M": "#2ca02c"}

# === Figure: Cell Line + Plate ===
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                c=[cl_colors[cl] for cl in cell_lines], **scatter_kw)
axes[0].set_title("lncRNA UMAP — Cell Line", fontsize=13)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
handles_cl = [plt.Line2D([0], [0], marker='o', color='w',
              markerfacecolor=cl_colors[cl], markersize=5, label=cl)
              for cl in unique_cls]
axes[0].legend(handles=handles_cl, title="Cell line",
               bbox_to_anchor=(1.05, 1), loc="upper left",
               fontsize=5, markerscale=1.5, ncol=3, title_fontsize=7)

axes[1].scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                c=[plate_colors[p] for p in plates], **scatter_kw)
axes[1].set_title("lncRNA UMAP — Plate (Batch)", fontsize=13)
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
handles_plate = [plt.Line2D([0], [0], marker='o', color='w',
                 markerfacecolor=plate_colors[p], markersize=6, label=p)
                 for p in unique_plates]
axes[1].legend(handles=handles_plate, title="Plate",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7, markerscale=1.2)

plt.suptitle(f"UMAP of SCVI-normalized lncRNA expression ({len(valid_genes)} genes, {n_pcs} PCs)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_cellline_plate.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → umap_cellline_plate.pdf")

# === Figure: Cell Cycle Phase ===
fig, ax = plt.subplots(figsize=(10, 8))
phases = meta[PHASE_COL].values
ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
           c=[phase_colors.get(p, "gray") for p in phases], **scatter_kw)
ax.set_title("lncRNA UMAP — Cell Cycle Phase", fontsize=13)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
handles_phase = [plt.Line2D([0], [0], marker='o', color='w',
                 markerfacecolor=phase_colors[p], markersize=8, label=p)
                 for p in ["G1", "S", "G2M"]]
ax.legend(handles=handles_phase, title="Phase", loc="upper right",
          fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_cellcycle.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → umap_cellcycle.pdf")

# === Figure: Drug (top 20 drugs highlighted, rest gray) ===
fig, ax = plt.subplots(figsize=(12, 10))
drugs = meta[DRUG_COL].values
drug_counts = pd.Series(drugs).value_counts()

# Exclude DMSO, take top 20 non-DMSO drugs
top_drugs = [d for d in drug_counts.index if d != "DMSO_TF"][:20]
top_drug_colors = {d: all_colors[i % len(all_colors)] for i, d in enumerate(top_drugs)}

# Background: all other drugs in gray
is_top = np.array([d in top_drug_colors for d in drugs])
ax.scatter(umap_embedding[~is_top, 0], umap_embedding[~is_top, 1],
           c="lightgray", s=0.3, alpha=0.1, rasterized=True, zorder=1)

# Highlighted drugs
for drug in top_drugs:
    mask = drugs == drug
    if mask.sum() > 0:
        ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                   c=[top_drug_colors[drug]], s=1, alpha=0.4, rasterized=True,
                   zorder=2, label=drug)

ax.set_title("lncRNA UMAP — Top 20 Drugs (by cell count)", fontsize=13)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(title="Drug", bbox_to_anchor=(1.05, 1), loc="upper left",
          fontsize=7, markerscale=4, title_fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_drug_top20.pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → umap_drug_top20.pdf")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'=' * 60}")
print("DONE")
print(f"  lncRNA genes used:    {len(valid_genes)}")
print(f"  Cells in subset:      {subset.n_obs:,}")
print(f"  Cell lines:           {len(unique_cls)}")
print(f"  Plates:               {len(unique_plates)}")
print(f"  Drugs:                {len(set(drugs))}")
print(f"  PCs:                  {n_pcs}")
print(f"  Total time:           {total:.0f}s ({total/60:.1f} min)")
print(f"\nOutputs in {OUTPUT_DIR}:")
print(f"  confusion_plate_cellline.pdf/csv")
print(f"  heatmap_drug_cellline.pdf")
print(f"  umap_cellline_plate.pdf")
print(f"  umap_cellcycle.pdf")
print(f"  umap_drug_top20.pdf")
print(f"  lncrna_pca_50.npy (reusable)")
print(f"  lncrna_umap_embedding.npy")
print(f"  lncrna_exploratory_metadata.parquet")
print(f"{'=' * 60}")