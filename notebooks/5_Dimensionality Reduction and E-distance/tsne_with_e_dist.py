"""
UNPROCESSED from Claude
t-SNE only — recompute t-SNE, reuse existing NNMDS
====================================================
Assumes NNMDS is already saved at OUTPUT_DIR/nnmds_embedding.npy
and metadata at OUTPUT_DIR/metadata.parquet.

Steps:
  1. Merge parquet files
  2. Parse metadata + exclude plate 14
  3. PCA (50 components, cuML GPU)
  4. t-SNE (cuML GPU, strictly default parameters — matching the paper)
  5. Save tsne_embedding.npy
  6. Reload NNMDS + plot combined figures colored by E-distance, cell line, drug


"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import time

# ─────────────────────────────────────────────
# CONFIGURATION — same paths as dim_reduction_v2
# ─────────────────────────────────────────────
DATA_DIR      = "/home/a/aangelopa/Thesis/Data/diff_vision_chunks"
OUTPUT_DIR    = "/home/a/aangelopa/Thesis/Data/dim_reduction"
E_DIST_DIR    = "/home/aangelopa/Thesis/Data/e_distance"

N_PCS_TSNE    = 50
EXCLUDE_PLATE = "14"
N_PCS_NNMDS   = 128   # only used for plot title

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Merge parquet files
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging parquet files")
print("=" * 60)
t0 = time.time()

parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "group_*.parquet")))
print(f"  Found {len(parquet_files)} parquet files")

if len(parquet_files) == 0:
    raise FileNotFoundError(f"No group_*.parquet files found in {DATA_DIR}")

dfs = []
for f in parquet_files:
    dfs.append(pd.read_parquet(f))
scores_df = pd.concat(dfs, axis=0)
del dfs

n_dups = scores_df.index.duplicated().sum()
if n_dups > 0:
    print(f"  WARNING: {n_dups} duplicates — dropping")
    scores_df = scores_df[~scores_df.index.duplicated(keep="first")]

print(f"  Shape: {scores_df.shape}")

# ─────────────────────────────────────────────
# STEP 2: Parse metadata + exclude plate 14
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Metadata + exclude plate 14")
print("=" * 60)

split_index = scores_df.index.str.split(r"\s*\|\s*", expand=True)
metadata = pd.DataFrame({
    "drug":      split_index.get_level_values(0),
    "plate":     split_index.get_level_values(1),
    "cell_line": split_index.get_level_values(2),
}, index=scores_df.index)

mask      = metadata["plate"].astype(str) != EXCLUDE_PLATE
scores_df = scores_df[mask]
metadata  = metadata[mask]

print(f"  Comparisons after plate 14 exclusion: {len(metadata)}")
metadata.to_parquet(os.path.join(OUTPUT_DIR, "metadata.parquet"))

scores          = np.nan_to_num(scores_df.values.astype(np.float32), nan=0.0)
n_comparisons   = scores.shape[0]
del scores_df
print(f"  Step 1-2 took {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
# STEP 3: PCA (cuML GPU)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PCA (cuML GPU)")
print("=" * 60)
t1 = time.time()

import cupy as cp
import cuml
from cuml.decomposition import PCA as cuPCA

print(f"  cuML version: {cuml.__version__}")
print(f"  Fitting {N_PCS_TSNE} PCs on {n_comparisons} comparisons...")

scores_gpu = cp.array(scores)
pca        = cuPCA(n_components=N_PCS_TSNE)   # no random_state — not supported by cuML
pcs_tsne   = cp.asnumpy(pca.fit_transform(scores_gpu)).astype(np.float32)
del scores_gpu, scores

ev = cp.asnumpy(pca.explained_variance_ratio_).cumsum()
print(f"  Variance explained by {N_PCS_TSNE} PCs: {ev[-1]*100:.1f}%")
print(f"  PCA took {time.time()-t1:.1f}s")

# ─────────────────────────────────────────────
# STEP 4: t-SNE (cuML GPU, default parameters)
# ─────────────────────────────────────────────
# Paper: "TSNE function in the cuML package with default parameters"
# cuML defaults: perplexity=30, learning_rate=200, n_iter=1000,
#                method="barnes_hut", no random_state
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: t-SNE (cuML GPU — default parameters, matching paper)")
print("=" * 60)
t2 = time.time()

import torch
from cuml.manifold import TSNE as cuTSNE

print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Input: {pcs_tsne.shape}  (default params: perplexity=30, lr=200)")

tsne           = cuTSNE(n_components=2)   # strictly default — no random_state
tsne_embedding = cp.asnumpy(tsne.fit_transform(cp.array(pcs_tsne)))
del pcs_tsne

np.save(os.path.join(OUTPUT_DIR, "tsne_embedding.npy"), tsne_embedding)
print(f"  Saved → tsne_embedding.npy  shape: {tsne_embedding.shape}")
print(f"  t-SNE took {time.time()-t2:.1f}s")

# ─────────────────────────────────────────────
# STEP 5: Load NNMDS (already computed)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Loading existing NNMDS embedding")
print("=" * 60)

nnmds_path = os.path.join(OUTPUT_DIR, "nnmds_embedding.npy")
if not os.path.exists(nnmds_path):
    raise FileNotFoundError(
        f"NNMDS embedding not found at {nnmds_path}\n"
        f"Run dim_reduction_v2.py first to generate it."
    )
nnmds_embedding = np.load(nnmds_path)
print(f"  Loaded nnmds_embedding.npy  shape: {nnmds_embedding.shape}")

# ─────────────────────────────────────────────
# STEP 6: Plots
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Generating plots")
print("=" * 60)

scatter_kw = dict(s=1, alpha=0.4, rasterized=True)

def make_color_map(values):
    unique_vals = sorted(set(values))
    all_colors  = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_vals)}
    pt_colors = np.array([color_map[v] for v in values])
    return unique_vals, color_map, pt_colors


# ── Plot 1: colored by E-distance ─────────────────────────────────────────
# Loads all e_distance_*.parquet files and merges them into one table
edist_files = sorted(glob.glob(os.path.join(E_DIST_DIR, "e_distance_*.parquet")))
if len(edist_files) > 0:
    print(f"  Loading {len(edist_files)} E-distance files from {E_DIST_DIR}...")
    edist_df  = pd.concat([pd.read_parquet(f) for f in edist_files], axis=0)
    print(f"  Merged E-distance shape: {edist_df.shape}")
    print(f"  Columns: {list(edist_df.columns)}")

    # Align to metadata index
    edist_aligned = edist_df.reindex(metadata.index)
    edist_col     = edist_aligned.columns[0]
    edist_values  = edist_aligned[edist_col].fillna(0).values
    log_edist     = np.log1p(edist_values)
    n_matched     = (edist_aligned[edist_col].notna()).sum()
    print(f"  Matched {n_matched:,} / {len(metadata):,} comparisons")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc0 = axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                          c=log_edist, cmap="viridis", **scatter_kw)
    axes[0].set_title("t-SNE colored by E-distance", fontsize=13)
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    plt.colorbar(sc0, ax=axes[0], label="log$_{1+}$ E-distance")

    sc1 = axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                          c=log_edist, cmap="viridis", **scatter_kw)
    axes[1].set_title("NNMDS colored by E-distance", fontsize=13)
    axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")
    plt.colorbar(sc1, ax=axes[1], label="log$_{1+}$ E-distance")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure5_bc_edistance.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
else:
    print(f"  No e_distance_*.parquet files found in {E_DIST_DIR} — skipping E-distance plot")


# ── Plot 2: colored by cell line ──────────────────────────────────────────
print("  Plotting colored by cell line...")
unique_cls, cl_color, pt_colors_cl = make_color_map(metadata["cell_line"].values)

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_cl, **scatter_kw)
axes[0].set_title(f"t-SNE ({N_PCS_TSNE} PCs) — Cell Line", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                c=pt_colors_cl, **scatter_kw)
axes[1].set_title(f"NNMDS ({N_PCS_NNMDS} PCs) — Cell Line", fontsize=13)
axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")

handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=cl_color[cl], markersize=5, label=cl)
           for cl in unique_cls]
axes[1].legend(handles=handles, title="Cell line",
               bbox_to_anchor=(1.05, 1), loc="upper left",
               fontsize=5, markerscale=1.5,
               ncol=max(1, len(unique_cls) // 20))

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "embeddings_by_cellline.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")


# ── Plot 3: colored by drug ───────────────────────────────────────────────
print("  Plotting colored by drug...")
unique_drugs, drug_color, pt_colors_drug = make_color_map(metadata["drug"].values)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                c=pt_colors_drug, **scatter_kw)
axes[0].set_title(f"t-SNE ({N_PCS_TSNE} PCs) — Drug", fontsize=13)
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

axes[1].scatter(nnmds_embedding[:, 0], nnmds_embedding[:, 1],
                c=pt_colors_drug, **scatter_kw)
axes[1].set_title(f"NNMDS ({N_PCS_NNMDS} PCs) — Drug", fontsize=13)
axes[1].set_xlabel("NNMDS 1"); axes[1].set_ylabel("NNMDS 2")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "embeddings_by_drug.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print(f"  Comparisons     : {tsne_embedding.shape[0]}")
print(f"  t-SNE shape     : {tsne_embedding.shape}")
print(f"  NNMDS shape     : {nnmds_embedding.shape}")
print(f"  Total time      : {total:.0f}s ({total/60:.1f} min)")
print(f"  Outputs in      : {OUTPUT_DIR}")
print("=" * 60)