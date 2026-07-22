"""
Leiden Clustering — Three Gene Lists: lncRNA, Coding, All
==========================================================

For each of the 3 gene lists this script does:
  1. Load H5AD + SCVI model  (ONCE)
  2. Subsample N_CELLS_PER_LINE cells per cell line  (ONCE)
  3. Get SCVI normalized expression for the UNION of all genes  (ONCE)
  4. For each gene list:
       a. Subset columns from shared matrix  (cheap numpy)
       b. Z-score per gene
       c. PCA (50 components)
       d. Leiden clustering
       e. UMAP + t-SNE
       f. Confusion heatmaps with Ward hierarchical dendrograms
       g. Save PDFs + metadata parquet

SLURM: --mem=64G  --gres=gpu:1  --time=04:00:00  --cpus-per-task=4
"""

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scvi.hub
import torch
from sklearn.preprocessing import StandardScaler
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
BASE_DIR  = Path.home() / "Thesis"
DATA_DIR  = BASE_DIR / "Data"

SNAPSHOT_PATH = Path(
    "/home/a/aangelopa/Thesis/Data/tahoe_cache/"
    "models--vevotx--Tahoe-100M-SCVI-v1/snapshots/"
    "b5283a73fbbed812a95264ace360da538b20af89"
)
DATA_PATH  = SNAPSHOT_PATH / "adata.h5ad"
CACHE_DIR  = Path("/home/a/aangelopa/Thesis/Data/tahoe_cache")

# UPDATE THESE to your actual file names
GENE_LISTS = {
    "lncrna": DATA_DIR / "1_for_geneset_building"/ "lnc_only_genes_final.txt",
    "coding": DATA_DIR / "1_for_geneset_building"/ "coding_only_genes_final.txt",
    "all":    DATA_DIR / "1_for_geneset_building"/ "all_genes_final.txt",
}

DRUG_META_PATH = DATA_DIR / "drug_metadata.parquet"
MOA_COL        = "moa-fine"
OUTPUT_DIR     = BASE_DIR / "Data" / "2ndtry" / "leiden_threelists_res0.3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_LINE_COL = "Cell_Name_Vevo"
PLATE_COL     = "plate"
DRUG_COL      = "drug"
PHASE_COL     = "phase"
DMSO_LABEL    = "DMSO_TF"

# FIX 2: 10k cells/line. Raise to 15k-20k only if node has >=128 GB RAM.
N_CELLS_PER_LINE  = 10000
N_PCS             = 50
LEIDEN_RESOLUTION = 0.3  # FIX 7
N_NEIGHBORS       = 15    # FIX 5
UMAP_MIN_DIST     = 0.5   # FIX 5
TSNE_PERPLEXITY   = 50
LIBRARY_SIZE      = 10e4
LOG_EPS           = 1e-6
RANDOM_STATE      = 42


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_gene_list(path):
    with open(path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]
    print(f"    {len(genes):,} genes from {path.name}")
    return genes


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


def plot_embedding(ax, coords, colors, title, xlabel, ylabel, scatter_kw=None):
    kw = dict(s=0.8, alpha=0.35, rasterized=True, edgecolors="none")
    if scatter_kw:
        kw.update(scatter_kw)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, **kw)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def add_legend(ax, unique_vals, color_map, title, fontsize=4, markersize=5):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[v], markersize=markersize, label=v)
        for v in unique_vals
    ]
    ax.legend(handles=handles, title=title,
              bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=fontsize, markerscale=1.5,
              ncol=max(1, len(unique_vals) // 20),
              title_fontsize=fontsize + 1)


def save_confusion_heatmap(ct_norm, title, xlabel, ylabel,
                           out_path, figsize=None, xtick_fs=6, ytick_fs=7):
    """FIX 8: Ward hierarchical clustermap with dendrograms on both axes."""
    if figsize is None:
        figsize = (max(10, ct_norm.shape[1] * 0.35),
                   max(8,  ct_norm.shape[0] * 0.35))
    g = sns.clustermap(
        ct_norm,
        cmap="YlOrRd",
        linewidths=0.1,
        figsize=figsize,
        cbar_kws={"label": "Fraction of cells", "shrink": 0.7},
        xticklabels=True,
        yticklabels=True,
        method="ward",
        metric="euclidean",
        row_cluster=ct_norm.shape[0] > 1,
        col_cluster=ct_norm.shape[1] > 1,
    )
    g.ax_heatmap.set_title(title, fontsize=13, pad=10)
    g.ax_heatmap.set_xlabel(xlabel, fontsize=10)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=10)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=xtick_fs, rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=ytick_fs, rotation=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out_path.name}")


# ─────────────────────────────────────────────
# STEP 1: Load model + AnnData  (ONCE)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading SCVI model and AnnData")
print("=" * 60)
t0 = time.time()

hub    = scvi.hub.HubModel.pull_from_huggingface_hub(
    repo_name="vevotx/Tahoe-100M-SCVI-v1",
    cache_dir=CACHE_DIR,
)
model  = hub.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to_device(device)
print(f"  Model on: {device}")

print(f"  Loading AnnData from {DATA_PATH}...")
adata_full = anndata.read_h5ad(DATA_PATH)
print(f"  AnnData: {adata_full.n_obs:,} cells x {adata_full.n_vars:,} genes")

# ─────────────────────────────────────────────
# STEP 2: Subsample  (ONCE)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 2: Subsampling {N_CELLS_PER_LINE:,} cells per cell line")
print("=" * 60)

np.random.seed(RANDOM_STATE)
sampled_indices      = []
cell_lines_available = adata_full.obs[CELL_LINE_COL].unique()
print(f"  Cell lines: {len(cell_lines_available)}")

for cl in cell_lines_available:
    idx = np.where(adata_full.obs[CELL_LINE_COL] == cl)[0]
    n   = min(N_CELLS_PER_LINE, len(idx))
    sampled_indices.extend(np.random.choice(idx, size=n, replace=False).tolist())

adata         = adata_full[sampled_indices].copy()
shared_obs    = adata.obs.copy().reset_index(drop=True)
var_names_set = set(adata.var_names)
del adata_full
print(f"  Subsampled: {adata.n_obs:,} cells total")

# ─────────────────────────────────────────────
# STEP 3: Load gene lists + compute union
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Loading gene lists and computing union")
print("=" * 60)

gene_lists_filtered = {}
for list_name, path in GENE_LISTS.items():
    raw      = load_gene_list(path)
    filtered = [g for g in raw if g in var_names_set]
    gene_lists_filtered[list_name] = filtered
    print(f"  {list_name}: {len(filtered):,} / {len(raw):,} genes in AnnData")

union_genes      = sorted(set(g for gl in gene_lists_filtered.values() for g in gl))
union_gene_index = {g: i for i, g in enumerate(union_genes)}
n_union          = len(union_genes)
n_cells          = adata.n_obs
mem_gb           = n_cells * n_union * 4 / 1e9
print(f"\n  Union: {n_union:,} genes")
print(f"  Expression matrix will be: {n_cells:,} x {n_union:,} = {mem_gb:.1f} GB")

# ─────────────────────────────────────────────
# STEP 4: ONE model call for all genes  (FIX 1)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Single SCVI model pass for all genes  (FIX 1)")
print("=" * 60)
t4 = time.time()

print(f"  Requesting {n_cells:,} cells x {n_union:,} genes...")
expr_union = model.get_normalized_expression(
    adata,
    gene_list=union_genes,
    library_size=LIBRARY_SIZE,
    return_numpy=True,
)
# Log-transform in-place (same as vision_scores_from_expr in diff_vision_scores_all40)
expr_union = np.log(expr_union.astype(np.float32) + LOG_EPS)
print(f"  Done. Shape: {expr_union.shape}  took {time.time() - t4:.0f}s")
del adata   # AnnData raw counts no longer needed

# ─────────────────────────────────────────────
# STEPS 5-9: Loop over each gene list
# No more model calls from here — only cheap numpy/scanpy.
# ─────────────────────────────────────────────

for list_name, genes_in_list in gene_lists_filtered.items():

    print("\n" + "=" * 60)
    print(f"GENE LIST: {list_name.upper()}  ({len(genes_in_list):,} genes)")
    print("=" * 60)
    t_list = time.time()

    out_subdir = OUTPUT_DIR / list_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    if len(genes_in_list) < 10:
        print("  WARNING: too few genes — skipping")
        continue

    # STEP 5: Subset columns (just numpy indexing, essentially free)
    print(f"  STEP 5: Subset {len(genes_in_list):,} genes from shared matrix")
    col_idx  = [union_gene_index[g] for g in genes_in_list]
    expr_sub = expr_union[:, col_idx]

    # STEP 6: Z-score per gene (FIX 3)
    print("  STEP 6: Z-scoring per gene...")
    expr_scaled = StandardScaler().fit_transform(expr_sub)

    # STEP 7: PCA
    print(f"  STEP 7: PCA ({N_PCS} components)...")
    t7        = time.time()
    adata_tmp = anndata.AnnData(X=expr_scaled)
    adata_tmp.obs = shared_obs.copy()
    sc.tl.pca(adata_tmp, n_comps=N_PCS, svd_solver="arpack",
              random_state=RANDOM_STATE)
    pcs = adata_tmp.obsm["X_pca"]
    ev  = adata_tmp.uns["pca"]["variance_ratio"].cumsum()
    print(f"    Variance explained: {ev[-1]*100:.1f}%  ({time.time()-t7:.0f}s)")
    np.save(out_subdir / f"{list_name}_pca_{N_PCS}.npy", pcs)
    del expr_scaled

    # STEP 8: Neighbors + Leiden + UMAP + t-SNE
    print("  STEP 8: Neighbors, Leiden, UMAP, t-SNE...")
    t8 = time.time()
    adata_tmp.obsm["X_pca_custom"] = pcs   # FIX 4

    sc.pp.neighbors(adata_tmp, use_rep="X_pca_custom",
                    n_neighbors=N_NEIGHBORS)   # FIX 5

    LEIDEN_KEY = f"leiden_{list_name}"
    sc.tl.leiden(adata_tmp, key_added=LEIDEN_KEY,
                 resolution=LEIDEN_RESOLUTION, random_state=RANDOM_STATE)
    n_clusters = adata_tmp.obs[LEIDEN_KEY].nunique()
    print(f"    Leiden clusters: {n_clusters}")

    sc.tl.umap(adata_tmp, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    sc.tl.tsne(adata_tmp, use_rep="X_pca_custom",   # FIX 6
               perplexity=TSNE_PERPLEXITY,
               random_state=RANDOM_STATE, n_jobs=-1)
    print(f"    Embeddings done ({time.time()-t8:.0f}s)")

    umap_coords = adata_tmp.obsm["X_umap"]
    tsne_coords = adata_tmp.obsm["X_tsne"]

    # Save metadata
    meta_out             = adata_tmp.obs.copy()
    meta_out[LEIDEN_KEY] = adata_tmp.obs[LEIDEN_KEY].values
    meta_out["UMAP1"]    = umap_coords[:, 0]
    meta_out["UMAP2"]    = umap_coords[:, 1]
    meta_out["TSNE1"]    = tsne_coords[:, 0]
    meta_out["TSNE2"]    = tsne_coords[:, 1]
    meta_out.to_parquet(out_subdir / f"{list_name}_leiden_metadata.parquet")

    # Color maps
    clusters        = meta_out[LEIDEN_KEY].values
    unique_clusters = sorted(set(clusters), key=int)
    cmap20          = plt.cm.tab20(np.linspace(0, 1, 20))
    cluster_colors  = {c: cmap20[int(c) % 20] for c in unique_clusters}
    unique_cls, cl_color, _ = make_color_map(meta_out[CELL_LINE_COL].values)
    plates        = meta_out[PLATE_COL].values
    unique_plates = sorted(set(plates))
    pcmap         = plt.cm.Set3(np.linspace(0, 1, max(len(unique_plates), 12)))
    plate_colors  = {p: pcmap[i % len(pcmap)] for i, p in enumerate(unique_plates)}
    phase_colors  = {"G1": "#1f77b4", "S": "#ff7f0e", "G2M": "#2ca02c"}
    phases        = meta_out[PHASE_COL].values
    handles_phase = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=phase_colors[p], markersize=8, label=p)
        for p in ["G1", "S", "G2M"]
    ]
    skw = dict(s=0.8, alpha=0.35, rasterized=True, edgecolors="none")

    # STEP 9a: UMAP plots
    print("  STEP 9a: UMAP plots...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    plot_embedding(axes[0], umap_coords,
                   [cluster_colors[c] for c in clusters],
                   f"{list_name} UMAP - Leiden (n={n_clusters})",
                   "UMAP 1", "UMAP 2", skw)
    if n_clusters <= 50:
        add_legend(axes[0], unique_clusters, cluster_colors, "Cluster",
                   fontsize=6, markersize=6)
    plot_embedding(axes[1], umap_coords,
                   [cl_color[cl] for cl in meta_out[CELL_LINE_COL].values],
                   f"{list_name} UMAP - Cell Line", "UMAP 1", "UMAP 2", skw)
    add_legend(axes[1], unique_cls, cl_color, "Cell line")
    plt.suptitle(f"{list_name} | res={LEIDEN_RESOLUTION} | {N_PCS} PCs | "
                 f"{N_CELLS_PER_LINE:,} cells/line", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_subdir / "umap_leiden_cellline.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    plot_embedding(axes[0], umap_coords, [cluster_colors[c] for c in clusters],
                   f"Leiden (n={n_clusters})", "UMAP 1", "UMAP 2", skw)
    plot_embedding(axes[1], umap_coords, [plate_colors[p] for p in plates],
                   "Plate (Batch)", "UMAP 1", "UMAP 2", skw)
    add_legend(axes[1], unique_plates, plate_colors, "Plate", fontsize=7, markersize=6)
    plot_embedding(axes[2], umap_coords, [phase_colors.get(p, "gray") for p in phases],
                   "Cell Cycle Phase", "UMAP 1", "UMAP 2", skw)
    axes[2].legend(handles=handles_phase, title="Phase",
                   loc="upper right", fontsize=9, markerscale=1.5)
    plt.suptitle(f"{list_name} UMAP - Batch & Biology check", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_subdir / "umap_leiden_plate_cycle.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved umap_leiden_cellline.pdf, umap_leiden_plate_cycle.pdf")

    # STEP 9b: t-SNE plots (FIX 6)
    print("  STEP 9b: t-SNE plots...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    plot_embedding(axes[0], tsne_coords,
                   [cluster_colors[c] for c in clusters],
                   f"{list_name} t-SNE - Leiden (n={n_clusters})",
                   "t-SNE 1", "t-SNE 2", skw)
    if n_clusters <= 50:
        add_legend(axes[0], unique_clusters, cluster_colors, "Cluster",
                   fontsize=6, markersize=6)
    plot_embedding(axes[1], tsne_coords,
                   [cl_color[cl] for cl in meta_out[CELL_LINE_COL].values],
                   f"{list_name} t-SNE - Cell Line", "t-SNE 1", "t-SNE 2", skw)
    add_legend(axes[1], unique_cls, cl_color, "Cell line")
    plt.suptitle(f"{list_name} | t-SNE perp={TSNE_PERPLEXITY} | {N_PCS} PCs | "
                 f"{N_CELLS_PER_LINE:,} cells/line", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_subdir / "tsne_leiden_cellline.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    plot_embedding(axes[0], tsne_coords, [cluster_colors[c] for c in clusters],
                   f"Leiden (n={n_clusters})", "t-SNE 1", "t-SNE 2", skw)
    plot_embedding(axes[1], tsne_coords, [plate_colors[p] for p in plates],
                   "Plate (Batch)", "t-SNE 1", "t-SNE 2", skw)
    add_legend(axes[1], unique_plates, plate_colors, "Plate", fontsize=7, markersize=6)
    plot_embedding(axes[2], tsne_coords, [phase_colors.get(p, "gray") for p in phases],
                   "Cell Cycle Phase", "t-SNE 1", "t-SNE 2", skw)
    axes[2].legend(handles=handles_phase, title="Phase",
                   loc="upper right", fontsize=9, markerscale=1.5)
    plt.suptitle(f"{list_name} t-SNE - Batch & Biology check", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_subdir / "tsne_leiden_plate_cycle.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved tsne_leiden_cellline.pdf, tsne_leiden_plate_cycle.pdf")

    # STEP 9c: Confusion heatmaps with dendrograms (FIX 8)
    print("  STEP 9c: Confusion heatmaps...")

    def norm_crosstab(col_a, col_b):
        ct = pd.crosstab(col_a, col_b)
        return ct.div(ct.sum(axis=0), axis=1)

    save_confusion_heatmap(
        norm_crosstab(meta_out[LEIDEN_KEY], meta_out[CELL_LINE_COL]),
        f"{list_name} - Leiden vs Cell Lines (res={LEIDEN_RESOLUTION})",
        "Cell Line", "Leiden Cluster",
        out_subdir / "leiden_vs_cellline.pdf",
        figsize=(20, max(8, n_clusters * 0.35)), xtick_fs=5)

    save_confusion_heatmap(
        norm_crosstab(meta_out[LEIDEN_KEY], meta_out[PLATE_COL]),
        f"{list_name} - Leiden vs Plates", "Plate", "Leiden Cluster",
        out_subdir / "leiden_vs_plate.pdf",
        figsize=(10, max(8, n_clusters * 0.35)))

    save_confusion_heatmap(
        norm_crosstab(meta_out[LEIDEN_KEY], meta_out[PHASE_COL]),
        f"{list_name} - Leiden vs Cell Cycle", "Phase", "Leiden Cluster",
        out_subdir / "leiden_vs_cellcycle.pdf",
        figsize=(6, max(8, n_clusters * 0.35)))

    top_drugs  = (meta_out[meta_out[DRUG_COL] != DMSO_LABEL][DRUG_COL]
                  .value_counts().head(30).index.tolist())
    meta_drugs = meta_out[meta_out[DRUG_COL].isin(top_drugs)]
    if len(meta_drugs) > 0:
        save_confusion_heatmap(
            norm_crosstab(meta_drugs[LEIDEN_KEY], meta_drugs[DRUG_COL]),
            f"{list_name} - Leiden vs Top 30 Drugs", "Drug", "Leiden Cluster",
            out_subdir / "leiden_vs_drug_top30.pdf",
            figsize=(16, max(8, n_clusters * 0.35)), xtick_fs=6)

    if DRUG_META_PATH.exists():
        drug_meta          = pd.read_parquet(DRUG_META_PATH)
        drug_meta["drug"]  = drug_meta["drug"].str.strip()
        meta_moa           = meta_out.copy()
        meta_moa[DRUG_COL] = meta_moa[DRUG_COL].str.strip()
        meta_moa           = meta_moa.merge(
            drug_meta[["drug", MOA_COL]], left_on=DRUG_COL, right_on="drug",
            how="left", suffixes=("", "_meta"))
        meta_moa_known = meta_moa[
            meta_moa[MOA_COL].notna() & (meta_moa[MOA_COL] != "unclear")]
        print(f"    Cells with known MOA: {len(meta_moa_known):,}")
        if len(meta_moa_known) > 0:
            save_confusion_heatmap(
                norm_crosstab(meta_moa_known[LEIDEN_KEY], meta_moa_known[MOA_COL]),
                f"{list_name} - Leiden vs MOA", "Mechanism of Action", "Leiden Cluster",
                out_subdir / "leiden_vs_moa.pdf",
                figsize=(16, max(8, n_clusters * 0.35)), xtick_fs=6)
    else:
        print("    Drug metadata not found - skipping MOA plot")

    print(f"\n  {list_name.upper()} DONE in {time.time()-t_list:.0f}s")
    for f in sorted(out_subdir.iterdir()):
        print(f"    {f.name}")

    del adata_tmp, pcs, umap_coords, tsne_coords, meta_out

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print("ALL DONE")
print(f"  Total time : {total:.0f}s  ({total/60:.1f} min)")
print(f"  Outputs in : {OUTPUT_DIR}")
print("=" * 60)