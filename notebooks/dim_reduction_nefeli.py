"""
Pipeline: Parquet → Matrix → PCA → t-SNE + NNMDS

Input:  parquet files 
        rows = comparisons: 'drug | plate | cell_line'
        cols = gene sets (differential Vision scores)

Output: tsne_embedding.npy    
        nnmds_embedding.npy  
        metadata.parquet      
        embeddings_overview.pdf

Notes:
  - Plate 14 is excluded (replication of plate 6)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import time


# CONFIGURATION

DATA_DIR    = "/home/a/aangelopa/Thesis/Data/diff_vision_chunks"
OUTPUT_DIR  = "/home/a/aangelopa/Thesis/Data/dim_reduction"

N_PCS_TSNE   = 50       # PCs for t-SNE input
N_PCS_NNMDS  = 128      # PCs for NNMDS input
NNMDS_EPOCHS = 250
NNMDS_BATCH  = 512
NNMDS_LR     = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Plate to exclude for visualization (replication plate)
EXCLUDE_PLATE = "14"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. MERGE ALL PARQUET FILES
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging parquet files")
print("=" * 60)
t0 = time.time()

parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "group_*.parquet")))
print(f"  Found {len(parquet_files)} parquet files")

if len(parquet_files) == 0:
    raise FileNotFoundError(f"No group_*.parquet files found in {DATA_DIR}")

# Read and concatenate
dfs = []
for f in parquet_files:
    dfs.append(pd.read_parquet(f))
scores_df = pd.concat(dfs, axis=0)
del dfs

# Check for duplicates in index
n_dups = scores_df.index.duplicated().sum()
if n_dups > 0:
    print(f"  WARNING: {n_dups} duplicate indices found — dropping duplicates")
    scores_df = scores_df[~scores_df.index.duplicated(keep='first')]

print(f"  Merged shape     : {scores_df.shape}")
print(f"  Total comparisons: {scores_df.shape[0]}")
print(f"  Gene sets        : {scores_df.shape[1]}")
print(f"  NaN count        : {scores_df.isna().sum().sum()}")
print(f"  dtype            : {scores_df.dtypes.unique()}")


# 2. METADATA 

print("\n" + "=" * 60)
print("STEP 2: Parsing metadata")
print("=" * 60)

# Index format: 'drug | plate | cell_line'
split_index = scores_df.index.str.split(r"\s*\|\s*", expand=True)
metadata = pd.DataFrame({
    "drug":      split_index.get_level_values(0),
    "plate":     split_index.get_level_values(1),
    "cell_line": split_index.get_level_values(2),
}, index=scores_df.index)

print(f"  Unique drugs     : {metadata['drug'].nunique()}")
print(f"  Unique plates    : {metadata['plate'].nunique()}")
print(f"  Unique cell lines: {metadata['cell_line'].nunique()}")
print(f"  Plates present   : {sorted(metadata['plate'].unique())}")


# 3. EXCLUDE PLATE 14 

print("\n" + "=" * 60)
print("STEP 3: Excluding replication plate")
print("=" * 60)

mask = metadata["plate"].astype(str) != EXCLUDE_PLATE
n_before = len(metadata)
scores_df = scores_df[mask]
metadata = metadata[mask]
n_after = len(metadata)

print(f"  Before: {n_before} comparisons")
print(f"  After:  {n_after} comparisons  (excluded {n_before - n_after})")

# Save metadata
meta_out = os.path.join(OUTPUT_DIR, "metadata.parquet")
metadata.to_parquet(meta_out)
print(f"  Metadata saved → {meta_out}")

# Save gene set names 
geneset_out = os.path.join(OUTPUT_DIR, "gene_set_names.txt")
with open(geneset_out, "w") as f:
    for name in scores_df.columns:
        f.write(name + "\n")
print(f"  Gene set names saved → {geneset_out}")


scores = np.nan_to_num(scores_df.values.astype(np.float32), nan=0.0)
n_comparisons, n_genesets = scores.shape
del scores_df
print(f"  Matrix: {n_comparisons} comparisons × {n_genesets} gene sets")

print(f"\n  Step 1-3 took {time.time() - t0:.1f}s")


# 4. PCA (cuML, GPU-accelerated)

print("\n" + "=" * 60)
print("STEP 4: PCA (cuML GPU)")
print("=" * 60)
t1 = time.time()

import cupy as cp
import cuml
from cuml.decomposition import PCA as cuPCA

print(f"  cuML version : {cuml.__version__}")

n_pcs = max(N_PCS_TSNE, N_PCS_NNMDS)
print(f"  Fitting PCA with {n_pcs} components on GPU...")

scores_gpu = cp.array(scores)
pca = cuPCA(n_components=n_pcs, random_state=42)
pcs_all = cp.asnumpy(pca.fit_transform(scores_gpu))
del scores_gpu, scores

ev = cp.asnumpy(pca.explained_variance_ratio_).cumsum()
print(f"  Variance explained by {N_PCS_TSNE}  PCs: {ev[N_PCS_TSNE-1]*100:.1f}%")
print(f"  Variance explained by {N_PCS_NNMDS} PCs: {ev[N_PCS_NNMDS-1]*100:.1f}%")


np.save(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"), ev)

pcs_tsne  = pcs_all[:, :N_PCS_TSNE].astype(np.float32)
pcs_nnmds = pcs_all[:, :N_PCS_NNMDS].astype(np.float32)
del pcs_all

print(f"  PCA took {time.time() - t1:.1f}s")


# 5. t-SNE (cuML GPU)

print("\n" + "=" * 60)
print("STEP 5: t-SNE (cuML GPU)")
print("=" * 60)
t2 = time.time()

from cuml.manifold import TSNE as cuTSNE

print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Input: {pcs_tsne.shape[0]} samples × {pcs_tsne.shape[1]} PCs")
print("  Running t-SNE with default parameters...")

tsne = cuTSNE(n_components=2, random_state=42)
tsne_embedding = cp.asnumpy(tsne.fit_transform(cp.array(pcs_tsne)))
del pcs_tsne

np.save(os.path.join(OUTPUT_DIR, "tsne_embedding.npy"), tsne_embedding)
print(f"  Saved → tsne_embedding.npy  shape: {tsne_embedding.shape}")
print(f"  t-SNE took {time.time() - t2:.1f}s")


# 6. NNMDS 

print("\n" + "=" * 60)
print("STEP 6: NNMDS (Neural Network MDS)")
print("=" * 60)
t3 = time.time()


class NNMDS(nn.Module):
    """
    (Canzar et al. 2024).
    Hidden layer width = 10 * input_dim 
    """
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        hidden = 10 * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def stress_loss(y_pred, x_batch):
    """Normalized stress: sum((d_in - d_out)^2) / sum(d_in^2)"""
    d_in  = torch.cdist(x_batch, x_batch)
    d_out = torch.cdist(y_pred,  y_pred)
    mask  = torch.triu(torch.ones_like(d_in, dtype=torch.bool), diagonal=1)
    d_in_flat  = d_in[mask]
    d_out_flat = d_out[mask]
    return ((d_in_flat - d_out_flat) ** 2).sum() / (d_in_flat ** 2).sum().clamp(min=1e-8)


print(f"  Device     : {DEVICE}")
print(f"  Input dim  : {N_PCS_NNMDS}  |  Hidden: {10 * N_PCS_NNMDS}  |  Output: 2")
print(f"  Epochs     : {NNMDS_EPOCHS}  |  Batch: {NNMDS_BATCH}  |  LR: {NNMDS_LR}")
print(f"  Training samples: {pcs_nnmds.shape[0]}")

X_tensor = torch.tensor(pcs_nnmds)
loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=NNMDS_BATCH,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
    drop_last=False,
)

model = NNMDS(N_PCS_NNMDS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=NNMDS_LR)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")

loss_history = []
print("\n  Training...")
for epoch in range(1, NNMDS_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch_x,) in loader:
        batch_x = batch_x.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = stress_loss(y_pred, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg = epoch_loss / n_batches
    loss_history.append(avg)
    if epoch % 25 == 0 or epoch == 1:
        print(f"    Epoch {epoch:>3}/{NNMDS_EPOCHS}  stress = {avg:.6f}")

# Generate final embeddings
model.eval()
with torch.no_grad():
    nnmds_parts = []
    for (batch_x,) in DataLoader(TensorDataset(X_tensor), batch_size=2048):
        nnmds_parts.append(model(batch_x.to(DEVICE)).cpu().numpy())
    nnmds_embedding = np.concatenate(nnmds_parts, axis=0)
del pcs_nnmds, X_tensor

np.save(os.path.join(OUTPUT_DIR, "nnmds_embedding.npy"), nnmds_embedding)
np.save(os.path.join(OUTPUT_DIR, "nnmds_loss.npy"), np.array(loss_history))
print(f"  Saved → nnmds_embedding.npy  shape: {nnmds_embedding.shape}")
print(f"  NNMDS took {time.time() - t3:.1f}s")


# 7. VISUALIZATION

print("\n" + "=" * 60)
print("STEP 7: Generating plots")
print("=" * 60)
 
metadata = pd.read_parquet(os.path.join(OUTPUT_DIR, "metadata.parquet"))
 
# color map
def make_color_map(values):
    unique_vals = sorted(set(values))
    n = len(unique_vals)
    all_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_vals)}
    pt_colors = np.array([color_map[v] for v in values])
    return unique_vals, color_map, pt_colors
 
#  by CELL LINE
print("  Plotting by cell line...")
unique_cls, cl_color, pt_colors_cl = make_color_map(metadata["cell_line"].values)
 
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
 
axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_cl, s=1, alpha=0.4, rasterized=True)
axes[0].set_title(f"t-SNE ({N_PCS_TSNE} PCs) — Cell Line", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
 
axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                c=pt_colors_cl, s=1, alpha=0.4, rasterized=True)
axes[1].set_title(f"NNMDS ({N_PCS_NNMDS} PCs) — Cell Line", fontsize=13)
axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")
 
if len(unique_cls) <= 60:
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cl_color[cl], markersize=5, label=cl)
               for cl in unique_cls]
    axes[1].legend(handles=handles, title="Cell line",
                   bbox_to_anchor=(1.05, 1), loc="upper left",
                   fontsize=5, markerscale=1.5, ncol=max(1, len(unique_cls) // 20))
 
axes[2].plot(range(1, NNMDS_EPOCHS + 1), loss_history, linewidth=1.5)
axes[2].set_title("NNMDS Training Loss", fontsize=13)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Normalized Stress")
axes[2].grid(True, alpha=0.4)
 
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "embeddings_by_cellline.pdf")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out1}")
 
#  by DRUG 
print("  Plotting by drug...")
unique_drugs, drug_color, pt_colors_drug = make_color_map(metadata["drug"].values)
 
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
 
axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_drug, s=1, alpha=0.4, rasterized=True)
axes[0].set_title(f"t-SNE ({N_PCS_TSNE} PCs) — Drug", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
 
axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                c=pt_colors_drug, s=1, alpha=0.4, rasterized=True)
axes[1].set_title(f"NNMDS ({N_PCS_NNMDS} PCs) — Drug", fontsize=13)
axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")
 
# Drug legend only if manageable
if len(unique_drugs) <= 40:
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=drug_color[d], markersize=5, label=d)
               for d in unique_drugs]
    axes[1].legend(handles=handles, title="Drug",
                   bbox_to_anchor=(1.05, 1), loc="upper left",
                   fontsize=4, markerscale=1.5, ncol=max(1, len(unique_drugs) // 25))
 
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "embeddings_by_drug.pdf")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out2}")
 

# SUMMARY

total_time = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  Comparisons         : {tsne_embedding.shape[0]}")
print(f"  Gene sets           : {n_genesets}")
print(f"  tsne_embedding.npy  : {tsne_embedding.shape}")
print(f"  nnmds_embedding.npy : {nnmds_embedding.shape}")
print(f"  metadata.parquet    : {metadata.shape}")
print(f"  Total time          : {total_time:.0f}s ({total_time/60:.1f} min)")
print("=" * 60)
