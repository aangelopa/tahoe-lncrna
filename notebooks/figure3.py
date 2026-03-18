"""
Figure 3 A, B, C: t-SNE of SCVI latent space
=============================================
A) Colored by cell line identity
B) Colored by plate (batch)
C) Colored by cell cycle phase

Method:
  - Load minified Tahoe SCVI model
  - Subsample 200,000 cells with equal contribution from top 47 cell lines
  - Get 10D latent representations
  - Run cuML t-SNE with default parameters
  - Plot three panels

Output:
  figure3_abc.pdf
  figure3_tsne_embedding.npy
  figure3_metadata.parquet
"""

import numpy as np
import pandas as pd
import torch
import anndata
import scvi.hub
import cupy as cp
from cuml.manifold import TSNE as cuTSNE
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
OUTPUT_DIR = BASE_DIR / "Data" 

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL = "plate"
PHASE_COL = "phase"

N_CELLS_TOTAL = 200_000
N_TOP_CELL_LINES = 47
BATCH_SIZE = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD MODEL AND DATA
# ─────────────────────────────────────────────
t0 = time.time()
print("=" * 60)
print("STEP 1: Loading model and data")
print("=" * 60)

tahoe_hubmodel = scvi.hub.HubModel.pull_from_huggingface_hub(
    repo_name="vevotx/Tahoe-100M-SCVI-v1",
    cache_dir=CACHE_DIR,
)
tahoe = tahoe_hubmodel.model
tahoe.to_device(DEVICE)
print(f"  Model on: {DEVICE}")

adata = anndata.read_h5ad(DATA_PATH)
print(f"  Full adata: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Filter to full-pass cells only
if "pass_filter" in adata.obs.columns:
    adata = adata[adata.obs["pass_filter"] == "full"].copy()
    print(f"  After pass_filter='full': {adata.n_obs:,} cells")

print(f"  Loading took {time.time() - t0:.0f}s")

# ─────────────────────────────────────────────
# 2. SUBSAMPLE: equal cells per top cell lines
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Subsampling cells")
print("=" * 60)

# Get top N cell lines by abundance
cl_counts = adata.obs[CELL_LINE_COL].value_counts()
top_cls = cl_counts.head(N_TOP_CELL_LINES).index.tolist()
print(f"  Top {N_TOP_CELL_LINES} cell lines selected")
print(f"  Smallest top cell line: {cl_counts[top_cls[-1]]:,} cells")

# Equal contribution from each cell line
cells_per_cl = N_CELLS_TOTAL // N_TOP_CELL_LINES
print(f"  Sampling {cells_per_cl} cells per cell line ({cells_per_cl * N_TOP_CELL_LINES:,} total)")

np.random.seed(42)
sampled_indices = []

for cl in top_cls:
    cl_mask = adata.obs[CELL_LINE_COL] == cl
    cl_indices = np.where(cl_mask)[0]
    n_available = len(cl_indices)

    if n_available >= cells_per_cl:
        chosen = np.random.choice(cl_indices, cells_per_cl, replace=False)
    else:
        # If fewer cells than needed, take all
        chosen = cl_indices
        print(f"  WARNING: {cl} has only {n_available} cells (need {cells_per_cl})")

    sampled_indices.append(chosen)

sampled_indices = np.concatenate(sampled_indices)
np.random.shuffle(sampled_indices)

subset = adata[sampled_indices].copy()
print(f"  Subset: {subset.n_obs:,} cells")

# ─────────────────────────────────────────────
# 3. GET LATENT REPRESENTATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing latent representations")
print("=" * 60)
t1 = time.time()

latent = tahoe.get_latent_representation(subset, batch_size=BATCH_SIZE)
print(f"  Latent shape: {latent.shape}")
print(f"  Took {time.time() - t1:.0f}s")

# ─────────────────────────────────────────────
# 4. t-SNE (cuML GPU)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: t-SNE (cuML GPU)")
print("=" * 60)
t2 = time.time()

print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Running t-SNE on {latent.shape[0]:,} cells × {latent.shape[1]} dims...")

tsne = cuTSNE(n_components=2, random_state=42)
tsne_embedding = cp.asnumpy(tsne.fit_transform(cp.array(latent.astype(np.float32))))

print(f"  t-SNE shape: {tsne_embedding.shape}")
print(f"  Took {time.time() - t2:.0f}s")

# Save embedding and metadata
np.save(OUTPUT_DIR / "figure3_tsne_embedding.npy", tsne_embedding)

meta = subset.obs[[CELL_LINE_COL, PLATE_COL, PHASE_COL]].copy()
meta.to_parquet(OUTPUT_DIR / "figure3_metadata.parquet")

# ─────────────────────────────────────────────
# 5. PLOT FIGURE 3 A, B, C
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating plots")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# --- Common plot settings ---
scatter_kw = dict(s=0.5, alpha=0.3, rasterized=True, edgecolors="none")

# --- Panel A: Colored by cell line ---
cell_lines = meta[CELL_LINE_COL].values
unique_cls = sorted(set(cell_lines))
n_cls = len(unique_cls)

# Build colormap for many cell lines
all_colors = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])
cl_colors = {cl: all_colors[i % len(all_colors)] for i, cl in enumerate(unique_cls)}
pt_colors_cl = np.array([cl_colors[cl] for cl in cell_lines])

axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_cl, **scatter_kw)
axes[0].set_title("Tahoe-100M t-SNE by Cell Line", fontsize=13)
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")

# --- Panel B: Colored by plate ---
plates = meta[PLATE_COL].values
unique_plates = sorted(set(plates))
n_plates = len(unique_plates)

plate_cmap = plt.cm.Set3(np.linspace(0, 1, max(n_plates, 12)))
plate_colors = {p: plate_cmap[i % len(plate_cmap)] for i, p in enumerate(unique_plates)}
pt_colors_plate = np.array([plate_colors[p] for p in plates])

axes[1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_plate, **scatter_kw)
axes[1].set_title("Tahoe-100M t-SNE by Batch (Plate)", fontsize=13)
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")

# Plate legend
handles_plate = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plate_colors[p], markersize=6, label=p)
    for p in unique_plates
]
axes[1].legend(handles=handles_plate, title="Plate",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=6, markerscale=1.2, ncol=1)

# --- Panel C: Colored by cell cycle phase ---
phases = meta[PHASE_COL].values
phase_color_map = {"G1": "#1f77b4", "S": "#ff7f0e", "G2M": "#2ca02c"}
pt_colors_phase = np.array([phase_color_map.get(p, "gray") for p in phases])

axes[2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_phase, **scatter_kw)
axes[2].set_title("Tahoe-100M t-SNE by Cell Cycle Phase", fontsize=13)
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")

# Phase legend
handles_phase = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=phase_color_map[p], markersize=8, label=p)
    for p in ["G1", "S", "G2M"]
]
axes[2].legend(handles=handles_phase, title="Phase",
               loc="upper right", fontsize=9, markerscale=1.5)

plt.tight_layout()
out_path = OUTPUT_DIR / "figure3_abc.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'=' * 60}")
print("DONE")
print(f"  Cells plotted:   {tsne_embedding.shape[0]:,}")
print(f"  Cell lines:      {n_cls}")
print(f"  Plates:          {n_plates}")
print(f"  Total time:      {total:.0f}s ({total/60:.1f} min)")
print(f"{'=' * 60}")