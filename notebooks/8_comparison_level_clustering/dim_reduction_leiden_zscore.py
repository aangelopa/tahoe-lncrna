"""
Comparison-Level Leiden + t-SNE + NNMDS on Differential Vision Scores (z-scored)
====================================================================================
Combines:
  - The z-scoring fix (per gene set, before PCA).
  - The full t-SNE + NNMDS pipeline from dim_reduction_v2.py 
  - Leiden clustering
  - Confusion heatmaps vs cell_line, top-30 drugs, MOA, and cell cycle phase
    composition.



Input:  Directory of parquet files (group_XXXX.parquet), same as
        dim_reduction_v2.py
Output: comparison_tsne_embedding.npy
        comparison_nnmds_embedding.npy
        comparison_leiden_metadata.parquet
        comparison_leiden_vs_cellline.pdf
        comparison_leiden_vs_drug_top30.pdf
        comparison_leiden_vs_moa.pdf
        comparison_leiden_vs_cellcycle.pdf
        comparison_embeddings_overview.pdf
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import anndata
import scanpy as sc
import glob
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = "/home/a/aangelopa/Thesis/Data/diff_vision_chunks"
OUTPUT_DIR     = "/home/a/aangelopa/Thesis/Data/2ndtry/comparison_level_full"
DRUG_META_PATH = "/home/a/aangelopa/Thesis/Data/drug_metadata.parquet"

# Path to the same HubModel AnnData used to build the differential scores -
# we only read .obs (metadata) from this, no model call, no expression matrix.
SNAPSHOT_PATH = (
    "/home/a/aangelopa/Thesis/Data/tahoe_cache/"
    "models--vevotx--Tahoe-100M-SCVI-v1/snapshots/"
    "b5283a73fbbed812a95264ace360da538b20af89"
)
ADATA_PATH = os.path.join(SNAPSHOT_PATH, "adata.h5ad")

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL     = "plate"
DRUG_COL      = "drug"
PHASE_COL     = "phase"
MOA_COL       = "moa-fine"
EXCLUDE_PLATE = "14"

N_PCS_TSNE   = 50
N_PCS_NNMDS  = 128
N_PCS_LEIDEN = 50          # PCs used for the neighbor graph -> Leiden
LEIDEN_RESOLUTION = 0.5
N_NEIGHBORS  = 15
RANDOM_STATE = 42

NNMDS_EPOCHS = 250
NNMDS_BATCH  = 4096
NNMDS_LR     = 1e-3
NNMDS_N_LANDMARKS = 1000
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

USE_GPU_PCA_TSNE = True   # cuML for PCA/t-SNE, matching dim_reduction_v2.py

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. MERGE ALL PARQUET FILES + PARSE METADATA
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

split_index = scores_df.index.str.split(r"\s*\|\s*", expand=True)
metadata = pd.DataFrame({
    "drug":      split_index.get_level_values(0),
    "plate":     split_index.get_level_values(1),
    "cell_line": split_index.get_level_values(2),
}, index=scores_df.index)

mask = metadata["plate"].astype(str) != EXCLUDE_PLATE
scores_df = scores_df[mask]
metadata  = metadata[mask]

gene_set_names = list(scores_df.columns)
scores = np.nan_to_num(scores_df.values.astype(np.float32), nan=0.0)
del scores_df

print(f"  Comparisons: {scores.shape[0]:,}  |  Gene sets: {scores.shape[1]:,}")
print(f"  Unique drugs: {metadata['drug'].nunique()}  |  "
      f"Unique cell lines: {metadata['cell_line'].nunique()}")

# ─────────────────────────────────────────────
# 2. Z-SCORE PER GENE SET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Z-scoring per gene set before PCA")
print("=" * 60)

gs_mean = scores.mean(axis=0, keepdims=True)
gs_std  = scores.std(axis=0, keepdims=True) + 1e-8
scores_z = (scores - gs_mean) / gs_std
np.save(os.path.join(OUTPUT_DIR, "geneset_zscore_mean.npy"), gs_mean)
np.save(os.path.join(OUTPUT_DIR, "geneset_zscore_std.npy"), gs_std)
del scores

# ─────────────────────────────────────────────
# 3. PCA (cuML, GPU) — shared across t-SNE, NNMDS, Leiden
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PCA")
print("=" * 60)
t1 = time.time()

n_pcs = max(N_PCS_TSNE, N_PCS_NNMDS, N_PCS_LEIDEN)

if USE_GPU_PCA_TSNE:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    scores_gpu = cp.array(scores_z)
    pca = cuPCA(n_components=n_pcs)
    pcs_all = cp.asnumpy(pca.fit_transform(scores_gpu))
    ev = cp.asnumpy(pca.explained_variance_ratio_).cumsum()
    del scores_gpu
else:
    from sklearn.decomposition import PCA as skPCA
    pca = skPCA(n_components=n_pcs, random_state=RANDOM_STATE)
    pcs_all = pca.fit_transform(scores_z)
    ev = np.cumsum(pca.explained_variance_ratio_)

print(f"  Variance explained by {N_PCS_TSNE}  PCs: {ev[N_PCS_TSNE-1]*100:.1f}%")
print(f"  Variance explained by {N_PCS_NNMDS} PCs: {ev[N_PCS_NNMDS-1]*100:.1f}%")
np.save(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"), ev)

pcs_tsne   = pcs_all[:, :N_PCS_TSNE].astype(np.float32)
pcs_nnmds  = pcs_all[:, :N_PCS_NNMDS].astype(np.float32)
pcs_leiden = pcs_all[:, :N_PCS_LEIDEN].astype(np.float32)
del pcs_all, scores_z
print(f"  PCA took {time.time()-t1:.0f}s")

# ─────────────────────────────────────────────
# 4. t-SNE (cuML, GPU) — same convention as dim_reduction_v2.py
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: t-SNE")
print("=" * 60)
t2 = time.time()

if USE_GPU_PCA_TSNE:
    from cuml.manifold import TSNE as cuTSNE
    tsne = cuTSNE(n_components=2, random_state=RANDOM_STATE)
    tsne_embedding = cp.asnumpy(tsne.fit_transform(cp.array(pcs_tsne)))
else:
    from sklearn.manifold import TSNE as skTSNE
    tsne = skTSNE(n_components=2, random_state=RANDOM_STATE)
    tsne_embedding = tsne.fit_transform(pcs_tsne)

np.save(os.path.join(OUTPUT_DIR, "comparison_tsne_embedding.npy"), tsne_embedding)
print(f"  t-SNE took {time.time()-t2:.0f}s")
del pcs_tsne

# ─────────────────────────────────────────────
# 5. NNMDS — identical architecture/fixes to dim_reduction_v2.py
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: NNMDS (landmark-based, v2 fixes)")
print("=" * 60)
t3 = time.time()

pcs_mean = pcs_nnmds.mean(axis=0, keepdims=True)
pcs_std  = pcs_nnmds.std(axis=0, keepdims=True) + 1e-8
pcs_nnmds_norm = (pcs_nnmds - pcs_mean) / pcs_std
np.save(os.path.join(OUTPUT_DIR, "nnmds_pca_mean.npy"), pcs_mean)
np.save(os.path.join(OUTPUT_DIR, "nnmds_pca_std.npy"), pcs_std)


class NNMDS(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        hidden = 10 * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def landmark_stress_loss(y_pred, x_batch, y_landmarks, x_landmarks):
    d_in  = torch.cdist(x_batch, x_landmarks)
    d_out = torch.cdist(y_pred,  y_landmarks)
    return ((d_in - d_out) ** 2).sum() / (d_in ** 2).sum().clamp(min=1e-8)


X_tensor = torch.tensor(pcs_nnmds_norm)
N = X_tensor.shape[0]
loader = DataLoader(
    TensorDataset(X_tensor, torch.arange(N)),
    batch_size=NNMDS_BATCH, shuffle=True,
    pin_memory=(DEVICE == "cuda"), drop_last=False,
)

nnmds_model = NNMDS(N_PCS_NNMDS).to(DEVICE)
optimizer = torch.optim.Adam(nnmds_model.parameters(), lr=NNMDS_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NNMDS_EPOCHS, eta_min=1e-5
)

print(f"  Device: {DEVICE}  |  Landmarks: {NNMDS_N_LANDMARKS}  |  "
      f"Epochs: {NNMDS_EPOCHS}  |  Batch: {NNMDS_BATCH}")

loss_history = []
for epoch in range(1, NNMDS_EPOCHS + 1):
    nnmds_model.train()
    landmark_idx = torch.randperm(N)[:NNMDS_N_LANDMARKS]
    x_landmarks = X_tensor[landmark_idx].to(DEVICE)
    with torch.no_grad():
        y_landmarks = nnmds_model(x_landmarks).detach()

    epoch_loss, n_batches = 0.0, 0
    for (batch_x, _) in loader:
        batch_x = batch_x.to(DEVICE)
        optimizer.zero_grad()
        y_pred = nnmds_model(batch_x)
        loss = landmark_stress_loss(y_pred, batch_x, y_landmarks, x_landmarks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    scheduler.step()

    avg = epoch_loss / n_batches
    loss_history.append(avg)
    if epoch % 25 == 0 or epoch == 1:
        print(f"    Epoch {epoch:>3}/{NNMDS_EPOCHS}  stress={avg:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

nnmds_model.eval()
with torch.no_grad():
    parts = []
    for (batch_x, _) in DataLoader(TensorDataset(X_tensor, torch.arange(N)),
                                    batch_size=4096):
        parts.append(nnmds_model(batch_x.to(DEVICE)).cpu().numpy())
    nnmds_embedding = np.concatenate(parts, axis=0)

np.save(os.path.join(OUTPUT_DIR, "comparison_nnmds_embedding.npy"), nnmds_embedding)
np.save(os.path.join(OUTPUT_DIR, "nnmds_loss.npy"), np.array(loss_history))
print(f"  NNMDS took {time.time()-t3:.0f}s")
del pcs_nnmds, pcs_nnmds_norm, X_tensor

# ─────────────────────────────────────────────
# 6. LEIDEN CLUSTERING 
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Leiden clustering (community detection on k-NN graph)")
print("=" * 60)

t4 = time.time()

adata = anndata.AnnData(X=pcs_leiden)
adata.obs_names = metadata.index
adata.obs["drug"]      = metadata["drug"].values
adata.obs["plate"]     = metadata["plate"].values
adata.obs["cell_line"] = metadata["cell_line"].values
adata.obsm["X_pca_precomputed"] = pcs_leiden

sc.pp.neighbors(adata, use_rep="X_pca_precomputed", n_neighbors=N_NEIGHBORS)
sc.tl.leiden(adata, key_added="leiden_comp", resolution=LEIDEN_RESOLUTION,
             random_state=RANDOM_STATE,flavor="leidenalg")
n_clusters   = adata.obs["leiden_comp"].nunique()
n_cell_lines = adata.obs["cell_line"].nunique()
print(f"  Leiden clusters: {n_clusters}  (cell lines: {n_cell_lines}, "
      f"drugs: {adata.obs['drug'].nunique()})")
print(f"  Leiden took {time.time()-t4:.0f}s")

# ─────────────────────────────────────────────
# 7. CELL CYCLE PHASE COMPOSITION PER GROUP
#    (reads only adata.obs metadata - no expression, no model call)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Computing cell cycle phase composition per group")
print("=" * 60)

phase_comp = None
if os.path.exists(ADATA_PATH):
    print(f"  Reading obs metadata only from {ADATA_PATH}...")
    full_obs = anndata.read_h5ad(ADATA_PATH, backed="r").obs[
        [DRUG_COL, PLATE_COL, CELL_LINE_COL, PHASE_COL]
    ].copy()
    full_obs["plate"] = full_obs[PLATE_COL].astype(str)
    full_obs = full_obs[full_obs["plate"] != EXCLUDE_PLATE]

    # Fraction of cells in each phase, per (drug, plate, cell_line) group -
    # matches the exact grouping used to build your differential scores
    phase_counts = (
        full_obs.groupby([DRUG_COL, "plate", CELL_LINE_COL])[PHASE_COL]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    phase_counts.columns = [f"frac_{c}" for c in phase_counts.columns]
    phase_counts["dominant_phase"] = phase_counts.idxmax(axis=1).str.replace("frac_", "")
    phase_counts = phase_counts.reset_index()
    phase_counts["index_key"] = (
        phase_counts[DRUG_COL].astype(str) + " | " +
        phase_counts["plate"].astype(str) + " | " +
        phase_counts[CELL_LINE_COL].astype(str)
    )
    phase_comp = phase_counts.set_index("index_key")
    print(f"  Computed phase composition for {len(phase_comp):,} groups")
    del full_obs
else:
    print(f"  WARNING: {ADATA_PATH} not found — skipping cell cycle merge")

# ─────────────────────────────────────────────
# 8. MERGE MOA + PHASE, SAVE METADATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Merging MOA + phase composition, saving metadata")
print("=" * 60)

meta_out = adata.obs[["drug", "plate", "cell_line", "leiden_comp"]].copy()
meta_out["TSNE1"]  = tsne_embedding[:, 0]
meta_out["TSNE2"]  = tsne_embedding[:, 1]
meta_out["NNMDS1"] = nnmds_embedding[:, 0]
meta_out["NNMDS2"] = nnmds_embedding[:, 1]

if os.path.exists(DRUG_META_PATH):
    drug_meta = pd.read_parquet(DRUG_META_PATH)
    drug_meta["drug"] = drug_meta["drug"].str.strip()
    meta_out["drug_stripped"] = meta_out["drug"].str.strip()
    meta_out = meta_out.merge(
        drug_meta[["drug", MOA_COL]], left_on="drug_stripped", right_on="drug",
        how="left", suffixes=("", "_meta")
    )
    print(f"  Comparisons with known MOA: {meta_out[MOA_COL].notna().sum():,}")
else:
    print(f"  WARNING: {DRUG_META_PATH} not found — skipping MOA merge")
    meta_out[MOA_COL] = np.nan

if phase_comp is not None:
    meta_out["index_key"] = (
        meta_out["drug"].astype(str) + " | " +
        meta_out["plate"].astype(str) + " | " +
        meta_out["cell_line"].astype(str)
    )
    meta_out = meta_out.merge(
        phase_comp[["dominant_phase"] + [c for c in phase_comp.columns if c.startswith("frac_")]],
        left_on="index_key", right_index=True, how="left"
    )
    print(f"  Comparisons with phase composition: {meta_out['dominant_phase'].notna().sum():,}")
else:
    meta_out["dominant_phase"] = np.nan

meta_out.to_parquet(os.path.join(OUTPUT_DIR, "comparison_leiden_metadata.parquet"))

# ─────────────────────────────────────────────
# 9. CONFUSION HEATMAPS
#    (Ward hierarchical clustering used ONLY to order rows/cols for display)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Confusion heatmaps")
print("=" * 60)

def norm_crosstab(col_a, col_b):
    ct = pd.crosstab(col_a, col_b)
    return ct.div(ct.sum(axis=0), axis=1)

def save_confusion_heatmap(ct_norm, title, xlabel, ylabel, out_path,
                            figsize=None, xtick_fs=6, ytick_fs=7):
    if figsize is None:
        figsize = (max(10, ct_norm.shape[1] * 0.35), max(8, ct_norm.shape[0] * 0.35))
    g = sns.clustermap(
        ct_norm, cmap="YlOrRd", linewidths=0.1, figsize=figsize,
        cbar_kws={"label": "Fraction", "shrink": 0.7},
        xticklabels=True, yticklabels=True, method="ward", metric="euclidean",
        row_cluster=ct_norm.shape[0] > 1, col_cluster=ct_norm.shape[1] > 1,
    )
    g.ax_heatmap.set_title(title, fontsize=13, pad=10)
    g.ax_heatmap.set_xlabel(xlabel, fontsize=10)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=10)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=xtick_fs, rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=ytick_fs, rotation=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out_path}")

# --- vs cell_line (sanity check) ---
save_confusion_heatmap(
    norm_crosstab(meta_out["leiden_comp"], meta_out["cell_line"]),
    f"Leiden vs Cell Line (res={LEIDEN_RESOLUTION})", "Cell Line", "Leiden Cluster",
    os.path.join(OUTPUT_DIR, "comparison_leiden_vs_cellline.pdf"),
    figsize=(20, max(8, n_clusters * 0.35)), xtick_fs=5
)

# --- vs top 30 drugs ---
top_drugs = meta_out["drug"].value_counts().head(30).index.tolist()
meta_drugs = meta_out[meta_out["drug"].isin(top_drugs)]
if len(meta_drugs) > 0:
    save_confusion_heatmap(
        norm_crosstab(meta_drugs["leiden_comp"], meta_drugs["drug"]),
        "Leiden vs Top 30 Drugs", "Drug", "Leiden Cluster",
        os.path.join(OUTPUT_DIR, "comparison_leiden_vs_drug_top30.pdf"),
        figsize=(16, max(8, n_clusters * 0.35)), xtick_fs=6
    )

# --- vs MOA ---
meta_known = meta_out[meta_out[MOA_COL].notna() & (meta_out[MOA_COL] != "unclear")]
if len(meta_known) > 0:
    save_confusion_heatmap(
        norm_crosstab(meta_known["leiden_comp"], meta_known[MOA_COL]),
        "Leiden vs MOA", "Mechanism of Action", "Leiden Cluster",
        os.path.join(OUTPUT_DIR, "comparison_leiden_vs_moa.pdf"),
        figsize=(16, max(8, n_clusters * 0.35)), xtick_fs=6
    )

# --- vs cell cycle (dominant phase per group) ---
meta_phase = meta_out[meta_out["dominant_phase"].notna()]
if len(meta_phase) > 0:
    save_confusion_heatmap(
        norm_crosstab(meta_phase["leiden_comp"], meta_phase["dominant_phase"]),
        "Leiden vs Dominant Cell Cycle Phase", "Phase", "Leiden Cluster",
        os.path.join(OUTPUT_DIR, "comparison_leiden_vs_cellcycle.pdf"),
        figsize=(6, max(8, n_clusters * 0.35))
    )

# ─────────────────────────────────────────────
# 10. OVERVIEW PLOT: t-SNE + NNMDS colored by Leiden cluster
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Overview plot")
print("=" * 60)

def make_color_map(values):
    unique_vals = sorted(set(values), key=lambda v: (str(v)))
    all_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_vals)}
    return unique_vals, color_map

clusters = meta_out["leiden_comp"].values
unique_clusters, cluster_color = make_color_map(clusters)
pt_colors = np.array([cluster_color[c] for c in clusters])

fig, axes = plt.subplots(1, 3, figsize=(26, 7))
axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors, s=1, alpha=0.4, rasterized=True)
axes[0].set_title(f"t-SNE ({N_PCS_TSNE} PCs) — Leiden (n={n_clusters})", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                c=pt_colors, s=1, alpha=0.4, rasterized=True)
axes[1].set_title(f"NNMDS ({N_PCS_NNMDS} PCs) — Leiden (n={n_clusters})", fontsize=13)
axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")

axes[2].plot(range(1, NNMDS_EPOCHS + 1), loss_history, linewidth=1.5)
axes[2].set_title("NNMDS Training Loss", fontsize=13)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Normalized Stress")
axes[2].grid(True, alpha=0.4)

plt.tight_layout()
out_overview = os.path.join(OUTPUT_DIR, "comparison_embeddings_overview.pdf")
plt.savefig(out_overview, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {out_overview}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print(f"  Comparisons     : {adata.n_obs:,}")
print(f"  Gene sets       : {len(gene_set_names):,}")
print(f"  Leiden clusters : {n_clusters}  (cell lines: {n_cell_lines})")
print(f"  Total time      : {total:.0f}s ({total/60:.1f} min)")
print(f"  Outputs in      : {OUTPUT_DIR}")
print("=" * 60)