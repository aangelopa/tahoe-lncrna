"""
Figure 3 A, B, C: lncRNA-based t-SNE
=====================================
Instead of using the full scVI latent space (as in the paper),
this uses SCVI-normalized expression of only lncRNA genes to
show how cells separate based on lncRNA-related gene expression.

A) Colored by cell line identity
B) Colored by plate (batch)
C) Colored by cell cycle phase

Method:
  1. Load lncRNA gene sets → extract all unique genes
  2. Subsample 200,000 cells (equal from top 47 cell lines)
  3. get_normalized_expression(gene_list=lnc_genes) from SCVI model
  4. PCA on the normalized expression
  5. cuML t-SNE on the PCs

Output:
  figure3_lncrna_abc.pdf
  figure3_lncrna_tsne_embedding.npy
  figure3_lncrna_metadata.parquet
"""

import numpy as np
import pandas as pd
import torch
import anndata
import scvi.hub
import cupy as cp
from cuml.manifold import TSNE as cuTSNE
from cuml.decomposition import PCA as cuPCA
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUTPUT_DIR = BASE_DIR / "Data" / "figure3"

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL = "plate"
PHASE_COL = "phase"

N_CELLS_TOTAL = 200_000
N_TOP_CELL_LINES = 47
N_PCS = 50              # PCs for t-SNE input
LIBRARY_SIZE = 10e4
BATCH_SIZE = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD lncRNA GENES FROM GENE SETS
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
print(f"  Gene sets loaded: {n_sets}")
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

# Filter to full-pass cells
if "pass_filter" in adata.obs.columns:
    adata = adata[adata.obs["pass_filter"] == "full"].copy()
    print(f"  After pass_filter='full': {adata.n_obs:,} cells")

# Filter gene list to those in the model
valid_genes = [g for g in lnc_genes if g in adata.var_names]
print(f"  Valid lncRNA genes in model: {len(valid_genes)} / {len(lnc_genes)}")

if len(valid_genes) == 0:
    raise ValueError("No lncRNA genes found in adata.var_names!")

print(f"  Loading took {time.time() - t0:.0f}s")

# ─────────────────────────────────────────────
# 3. SUBSAMPLE CELLS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Subsampling cells")
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
# 4. GET NORMALIZED EXPRESSION (lncRNA genes only)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Getting SCVI-normalized expression for lncRNA genes")
print("=" * 60)
t1 = time.time()

print(f"  Generating expression for {len(valid_genes)} genes × {subset.n_obs:,} cells...")
expr = tahoe.get_normalized_expression(
    subset,
    gene_list=valid_genes,
    library_size=LIBRARY_SIZE,
    batch_size=BATCH_SIZE,
)

# Convert to numpy if DataFrame
if isinstance(expr, pd.DataFrame):
    expr_matrix = expr.values.astype(np.float32)
else:
    expr_matrix = np.array(expr, dtype=np.float32)

print(f"  Expression matrix: {expr_matrix.shape}")
print(f"  Took {time.time() - t1:.0f}s")

# ─────────────────────────────────────────────
# 5. PCA + t-SNE (GPU)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: PCA + t-SNE (cuML GPU)")
print("=" * 60)
t2 = time.time()

# PCA
n_pcs = min(N_PCS, expr_matrix.shape[1])
print(f"  PCA: {expr_matrix.shape[1]} genes → {n_pcs} PCs")
expr_gpu = cp.array(expr_matrix)
pca = cuPCA(n_components=n_pcs)
pcs = cp.asnumpy(pca.fit_transform(expr_gpu))
del expr_gpu, expr_matrix

ev = cp.asnumpy(pca.explained_variance_ratio_).cumsum()
print(f"  Variance explained by {n_pcs} PCs: {ev[-1]*100:.1f}%")

# t-SNE
print(f"  t-SNE on {pcs.shape[0]:,} cells × {pcs.shape[1]} PCs...")
tsne = cuTSNE(n_components=2, random_state=42)
tsne_embedding = cp.asnumpy(tsne.fit_transform(cp.array(pcs.astype(np.float32))))
del pcs

print(f"  t-SNE shape: {tsne_embedding.shape}")
print(f"  PCA + t-SNE took {time.time() - t2:.0f}s")

# Save
np.save(OUTPUT_DIR / "figure3_lncrna_tsne_embedding.npy", tsne_embedding)

meta = subset.obs[[CELL_LINE_COL, PLATE_COL, PHASE_COL]].copy()
meta.to_parquet(OUTPUT_DIR / "figure3_lncrna_metadata.parquet")

# ─────────────────────────────────────────────
# 6. PLOT FIGURE 3 A, B, C
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Generating plots")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
scatter_kw = dict(s=0.5, alpha=0.3, rasterized=True, edgecolors="none")

# --- Panel A: Cell line ---
cell_lines = meta[CELL_LINE_COL].values
unique_cls = sorted(set(cell_lines))
all_colors = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])
cl_colors = {cl: all_colors[i % len(all_colors)] for i, cl in enumerate(unique_cls)}
pt_cl = np.array([cl_colors[cl] for cl in cell_lines])

axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=pt_cl, **scatter_kw)
axes[0].set_title("lncRNA t-SNE — Cell Line", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

# --- Panel B: Plate ---
plates = meta[PLATE_COL].values
unique_plates = sorted(set(plates))
plate_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_plates), 12)))
plate_colors = {p: plate_cmap[i % len(plate_cmap)] for i, p in enumerate(unique_plates)}
pt_plate = np.array([plate_colors[p] for p in plates])

axes[1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=pt_plate, **scatter_kw)
axes[1].set_title("lncRNA t-SNE — Plate (Batch)", fontsize=13)
axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

handles_plate = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plate_colors[p], markersize=6, label=p)
    for p in unique_plates
]
axes[1].legend(handles=handles_plate, title="Plate",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=6, markerscale=1.2)

# --- Panel C: Cell cycle phase ---
phases = meta[PHASE_COL].values
phase_colors = {"G1": "#1f77b4", "S": "#ff7f0e", "G2M": "#2ca02c"}
pt_phase = np.array([phase_colors.get(p, "gray") for p in phases])

axes[2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=pt_phase, **scatter_kw)
axes[2].set_title("lncRNA t-SNE — Cell Cycle Phase", fontsize=13)
axes[2].set_xlabel("t-SNE 1"); axes[2].set_ylabel("t-SNE 2")

handles_phase = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=phase_colors[p], markersize=8, label=p)
    for p in ["G1", "S", "G2M"]
]
axes[2].legend(handles=handles_phase, title="Phase",
               loc="upper right", fontsize=9, markerscale=1.5)

plt.suptitle(
    f"t-SNE of SCVI-normalized expression ({len(valid_genes)} lncRNA genes, {n_pcs} PCs)",
    fontsize=14, y=1.02
)
plt.tight_layout()
out_path = OUTPUT_DIR / "figure3_lncrna_abc.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'=' * 60}")
print("DONE")
print(f"  lncRNA genes used:  {len(valid_genes)}")
print(f"  Cells plotted:      {tsne_embedding.shape[0]:,}")
print(f"  Cell lines:         {len(unique_cls)}")
print(f"  PCs for t-SNE:      {n_pcs}")
print(f"  Total time:         {total:.0f}s ({total/60:.1f} min)")
print(f"{'=' * 60}")