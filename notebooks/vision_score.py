"""
Vision Score Computation for Gene Set Activity
================================================
Computes gene set activity scores using an adapted Vision score:
    score = log(geometric_mean(normalized_expression_within_set))
           = mean(log(normalized_expression_within_set))

Applied to SCVI-normalized expression levels from the Tahoe model.
"""

import torch
import numpy as np
import pandas as pd
import anndata
import scvi.hub
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path.home() / "Thesis"
DATA_DIR      = BASE_DIR / "Data"
CACHE_DIR     = Path("/home/a/aangelopa/Thesis/Data/tahoe_cache")
SNAPSHOT_PATH = Path(
    "/home/a/aangelopa/Thesis/Data/tahoe_cache/"
    "models--vevotx--Tahoe-100M-SCVI-v1/snapshots/"
    "b5283a73fbbed812a95264ace360da538b20af89"
)
DATA_PATH          = SNAPSHOT_PATH / "adata.h5ad"
GENE_SET_FILE      = DATA_DIR / "scvi_hvgs_genesets_filtered_15-500-10_sorted.txt"  # your filtered gene sets
OUTPUT_SCORES      = DATA_DIR / "vision_scores.parquet"             # cell × gene-set matrix
LIBRARY_SIZE       = 10e4                                 # matches HVG_scvi02.py
LOG_PSEUDOCOUNT    = 1e-6                                 # added before log to avoid log(0)
BATCH_SIZE         = 5000                                 # cells per batch — tune based on available GPU memory


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cache_dir):
    print("Loading Tahoe SCVI model...")
    hub = scvi.hub.HubModel.pull_from_huggingface_hub(
        repo_name="vevotx/Tahoe-100M-SCVI-v1",
        cache_dir=cache_dir,
    )
    model = hub.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to_device(device)
    print(f"  Model loaded on: {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Load gene sets (tab-separated: Name \t Source \t gene1,gene2,...)
# ─────────────────────────────────────────────────────────────────────────────

def load_gene_sets(gene_set_file):
    print(f"Loading gene sets from {gene_set_file}...")
    gene_sets = {}
    with open(gene_set_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name  = parts[0]
            genes = [g.strip() for g in parts[2].split(",") if g.strip()]
            gene_sets[name] = genes
    print(f"  Loaded {len(gene_sets)} gene sets.")
    return gene_sets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Compute Vision scores
#
# Vision score for a gene set G in cell c:
#   score(c, G) = log( geometric_mean( expr(c, g) for g in G ) )
#               = mean( log( expr(c, g) + ε ) for g in G )
#
# We get normalized expression in one batch call per unique set of genes,
# then slice columns for each gene set.
# ─────────────────────────────────────────────────────────────────────────────

def compute_vision_scores(model, adata, gene_sets, library_size, log_eps, batch_size=5000):
    """
    Parameters
    ----------
    model       : loaded SCVI model
    adata       : AnnData object (cells × genes, same vocab as model)
    gene_sets   : dict {set_name: [gene, ...]}
    library_size: normalisation library size
    log_eps     : small constant to avoid log(0)

    Returns
    -------
    scores_df : pd.DataFrame, shape (n_cells, n_gene_sets)
    """

    # ── Collect all unique genes needed across all gene sets ──────────────────
    all_needed_genes = set()
    for genes in gene_sets.values():
        all_needed_genes.update(genes)

    # Restrict only to genes present in the model's vocabulary
    valid_genes = sorted(
        g for g in all_needed_genes
        if g in adata.var_names
    )
    print(f"  Unique genes needed across all sets: {len(all_needed_genes)}")
    print(f"  Valid (present in model): {len(valid_genes)}")

    gene_index = {g: i for i, g in enumerate(valid_genes)}
    set_names  = list(gene_sets.keys())

    # Pre-compute column indices for each gene set (only once)
    set_col_indices = {}
    for set_name in set_names:
        cols = [gene_index[g] for g in gene_sets[set_name] if g in gene_index]
        if cols:
            set_col_indices[set_name] = cols

    n_cells = adata.n_obs
    scores  = np.full((n_cells, len(set_names)), np.nan, dtype=np.float32)

    print(f"  Scoring {len(set_names)} gene sets across {n_cells} cells in batches of {batch_size}...")

    for batch_start in range(0, n_cells, batch_size):
        batch_end   = min(batch_start + batch_size, n_cells)
        batch_adata = adata[batch_start:batch_end].copy()

        # Get normalized expression for this batch only
        norm_expr_df = model.get_normalized_expression(
            batch_adata,
            gene_list=valid_genes,
            library_size=library_size,
        )
        log_expr = np.log(norm_expr_df.values + log_eps)  # (batch_size, n_valid_genes)

        # Score each gene set for this batch
        for j, set_name in enumerate(set_names):
            cols = set_col_indices.get(set_name)
            if cols is None:
                continue
            scores[batch_start:batch_end, j] = log_expr[:, cols].mean(axis=1)

        print(f"    ... processed cells {batch_end}/{n_cells}")

    scores_df = pd.DataFrame(
        scores,
        index=adata.obs_names,
        columns=set_names,
        dtype=np.float32,
    )
    return scores_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load model
    model = load_model(CACHE_DIR)

    # 2. Load AnnData and subsample to 10,000 cells
    print(f"Loading AnnData from {DATA_PATH}...")
    adata = anndata.read_h5ad(DATA_PATH)
    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

    print("Subsampling 10,000 random cells...")
    random_indices = np.random.choice(adata.n_obs, 10000, replace=False)
    adata = adata[random_indices].copy()
    print(f"  Subset shape: {adata.shape}")

    # 3. Load gene sets
    gene_sets = load_gene_sets(GENE_SET_FILE)
    # 4. Compute Vision scores
    print("\nComputing Vision scores...")
    scores_df = compute_vision_scores(
        model, adata, gene_sets,
        library_size=LIBRARY_SIZE,
        log_eps=LOG_PSEUDOCOUNT,
        batch_size=BATCH_SIZE,
    )

    # 5. Save
    print(f"\nSaving scores to {OUTPUT_SCORES}...")
    scores_df.to_parquet(OUTPUT_SCORES)
    print(f"Done! Matrix shape: {scores_df.shape}  (cells × gene sets)")
    print(scores_df.iloc[:5, :5])   # quick preview


if __name__ == "__main__":
    main()