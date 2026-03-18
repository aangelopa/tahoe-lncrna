"""
Differential Vision Score Computation — SLURM Array Version
=============================================================
Each SLURM job processes GROUPS_PER_JOB (plate, cell_line) groups.

Optimised MC loop — ONE model call per sample covers ALL groups in the job:

FOR each of 100 MC samples:
    1. For every group, subsample 100 cells per drug + 100 DMSO cells
    2. Concatenate ALL sampled cells across ALL groups into one batch
    3. ONE model call for the entire batch
    4. Compute Vision scores for the entire batch at once
    5. Slice back to each group, compute drug mean - DMSO mean
    6. Store differential scores for this sample

AFTER 100 samples:
    Median across samples → final scores per group

This gives 100 model calls per job instead of 40 groups × 100 = 4,000.

Usage:
    python differential_vision_scores_slurm.py --job_id 0
    python differential_vision_scores_slurm.py --merge
"""

import argparse
import torch
import numpy as np
import pandas as pd
import anndata
import scvi.hub
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path.home() / "Thesis"
DATA_DIR       = BASE_DIR / "Data"
CACHE_DIR      = Path("/home/a/aangelopa/Thesis/Data/tahoe_cache")
SNAPSHOT_PATH  = Path(
    "/home/a/aangelopa/Thesis/Data/tahoe_cache/"
    "models--vevotx--Tahoe-100M-SCVI-v1/snapshots/"
    "b5283a73fbbed812a95264ace360da538b20af89"
)
DATA_PATH      = SNAPSHOT_PATH / "adata.h5ad"
GENE_SET_FILE  = DATA_DIR / "scvi_hvgs_genesets_filtered_15-500-10_sorted.txt"
PARTIAL_DIR    = DATA_DIR / "diff_vision_chunks"
OUTPUT_SCORES  = DATA_DIR / "differential_vision_scores.parquet"

DMSO_LABEL     = "DMSO_TF"
CELL_LINE_COL  = "Cell_Name_Vevo"
PLATE_COL      = "plate"
DRUG_COL       = "drug"

MIN_CELLS      = 50
N_MC_SAMPLES   = 100
CELLS_PER_CAT  = 100     # cells sampled per drug/DMSO category per MC sample
GROUPS_PER_JOB = 40      # 674 groups / 40 = 17 jobs
LIBRARY_SIZE   = 10e4
LOG_EPS        = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
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


def load_gene_sets(gene_set_file, var_names):
    print(f"Loading gene sets from {gene_set_file}...")
    gene_sets     = {}
    var_names_set = set(var_names)
    with open(gene_set_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name  = parts[0]
            genes = [g.strip() for g in parts[2].split(",")
                     if g.strip() in var_names_set]
            if genes:
                gene_sets[name] = genes
    print(f"  Loaded {len(gene_sets):,} gene sets.")
    return gene_sets


def build_group_list(obs):
    """
    Return sorted list of (plate, cell_line) tuples with at least one
    valid drug comparison (drug >= MIN_CELLS and DMSO >= MIN_CELLS).
    """
    valid_groups = []
    for (plate, cell_line), grp in obs.groupby(
        [PLATE_COL, CELL_LINE_COL], observed=True
    ):
        dmso_count  = (grp[DRUG_COL] == DMSO_LABEL).sum()
        drug_counts = (
            grp[grp[DRUG_COL] != DMSO_LABEL]
            .groupby(DRUG_COL, observed=True)
            .size()
        )
        valid_drugs = drug_counts[drug_counts >= MIN_CELLS]
        if dmso_count >= MIN_CELLS and len(valid_drugs) > 0:
            valid_groups.append((plate, cell_line))
    return sorted(valid_groups)


def vision_scores_from_expr(norm_expr, gene_sets, gene_index, log_eps):
    """
    norm_expr  : np.ndarray (n_cells, n_genes)
    Returns    : np.ndarray (n_cells, n_gene_sets)
    """
    log_expr  = np.log(norm_expr.astype(np.float32) + log_eps)
    set_names = list(gene_sets.keys())
    scores    = np.full(
        (norm_expr.shape[0], len(set_names)), np.nan, dtype=np.float32
    )
    for j, name in enumerate(set_names):
        cols = [gene_index[g] for g in gene_sets[name] if g in gene_index]
        if cols:
            scores[:, j] = log_expr[:, cols].mean(axis=1)
    return scores, set_names


# ─────────────────────────────────────────────────────────────────────────────
# CORE: process one job (GROUPS_PER_JOB groups, 100 model calls total)
# ─────────────────────────────────────────────────────────────────────────────

def process_job(job_id):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model, AnnData, gene sets once per job ───────────────────────────
    model = load_model(CACHE_DIR)

    print(f"Loading AnnData from {DATA_PATH}...")
    adata = anndata.read_h5ad(DATA_PATH)
    print(f"  AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    gene_sets  = load_gene_sets(GENE_SET_FILE, adata.var_names)
    all_genes  = sorted({g for genes in gene_sets.values() for g in genes})
    gene_index = {g: i for i, g in enumerate(all_genes)}
    set_names  = list(gene_sets.keys())
    n_sets     = len(set_names)

    obs          = adata.obs[[DRUG_COL, PLATE_COL, CELL_LINE_COL]].copy()
    valid_groups = build_group_list(obs)
    total_groups = len(valid_groups)

    # Slice the groups assigned to this job
    start_group = job_id * GROUPS_PER_JOB
    end_group   = min(start_group + GROUPS_PER_JOB, total_groups)

    if start_group >= total_groups:
        print(f"Job {job_id}: start {start_group} >= total {total_groups}. Nothing to do.")
        return

    print(f"\nJob {job_id}: groups {start_group}–{end_group - 1} "
          f"({end_group - start_group} groups)")

    # ── Identify which groups still need processing ───────────────────────────
    pending = []
    for group_id in range(start_group, end_group):
        partial_path = PARTIAL_DIR / f"group_{group_id:04d}.parquet"
        if partial_path.exists():
            print(f"  Group {group_id}: already done, skipping.")
        else:
            pending.append(group_id)

    if not pending:
        print("All groups in this job already done.")
        return

    print(f"  Groups to process: {len(pending)}")

    # ── Build per-group metadata (cell positions, drug names etc.) ────────────
    # This is computed once here and reused across all 100 MC samples
    print("\nPreparing group metadata...")
    group_meta = []   # list of dicts, one per pending group

    for group_id in pending:
        plate, cell_line = valid_groups[group_id]

        group_mask  = (obs[PLATE_COL] == plate) & (obs[CELL_LINE_COL] == cell_line)
        group_cells = obs.index[group_mask].tolist()
        group_obs   = obs.loc[group_cells]
        cell_to_pos = {cell: i for i, cell in enumerate(group_cells)}

        dmso_cells  = group_obs.index[group_obs[DRUG_COL] == DMSO_LABEL].tolist()
        dmso_pos    = [cell_to_pos[c] for c in dmso_cells]

        drug_groups = (
            group_obs[group_obs[DRUG_COL] != DMSO_LABEL]
            .groupby(DRUG_COL, observed=True)
        )
        valid_drugs = {
            drug: [cell_to_pos[c] for c in cells.index.tolist()]
            for drug, cells in drug_groups
            if len(cells) >= MIN_CELLS
        }

        if not valid_drugs:
            print(f"  Group {group_id} (plate={plate}, {cell_line}): no valid comparisons, skipping.")
            continue

        drug_names   = sorted(valid_drugs.keys())
        n_comp       = len(drug_names)

        # Load this group's AnnData subset (kept in memory for the whole job)
        adata_group  = adata[group_cells].copy()

        group_meta.append({
            "group_id":   group_id,
            "plate":      plate,
            "cell_line":  cell_line,
            "adata":      adata_group,
            "dmso_pos":   dmso_pos,
            "drug_names": drug_names,
            "drug_pos":   valid_drugs,
            "n_comp":     n_comp,
            # Allocate MC storage here: (n_samples, n_drugs, n_sets)
            "mc_diff":    np.full((N_MC_SAMPLES, n_comp, n_sets), np.nan, dtype=np.float32),
        })

        print(f"  Group {group_id}: plate={plate}, {cell_line} | "
              f"{len(group_cells):,} cells | {n_comp} drugs")

    if not group_meta:
        print("No valid groups to process.")
        return

    # ── MC LOOP: 100 iterations, ONE model call per iteration ─────────────────
    print(f"\nStarting {N_MC_SAMPLES} MC samples across {len(group_meta)} groups...")
    print(f"  One model call per sample covers all groups simultaneously.")

    for s in range(N_MC_SAMPLES):
        if s % 10 == 0:
            print(f"  Sample {s + 1}/{N_MC_SAMPLES}...")

        # STEP 1: Subsample cells for ALL groups and concatenate into one batch
        # Track which rows in the batch belong to which (group, category)
        batch_indices = []   # global AnnData indices (barcodes) for model call
        batch_meta    = []   # (group_idx, 'dmso'|drug_name, start, end)

        for g_idx, gm in enumerate(group_meta):
            # Subsample DMSO
            n_dmso      = min(CELLS_PER_CAT, len(gm["dmso_pos"]))
            dmso_sample = np.random.choice(
                gm["dmso_pos"], size=n_dmso, replace=False
            ).tolist()
            start = len(batch_indices)
            # Convert local positions back to global AnnData barcodes
            local_barcodes = gm["adata"].obs_names.tolist()
            batch_indices.extend([local_barcodes[i] for i in dmso_sample])
            batch_meta.append((g_idx, "dmso", start, len(batch_indices), n_dmso))

            # Subsample each drug
            for drug in gm["drug_names"]:
                n_drug      = min(CELLS_PER_CAT, len(gm["drug_pos"][drug]))
                drug_sample = np.random.choice(
                    gm["drug_pos"][drug], size=n_drug, replace=False
                ).tolist()
                start = len(batch_indices)
                batch_indices.extend([local_barcodes[i] for i in drug_sample])
                batch_meta.append((g_idx, drug, start, len(batch_indices), n_drug))

        # STEP 2: ONE model call for the entire batch across all groups
        adata_batch = adata[batch_indices]
        expr_all    = model.get_normalized_expression(
            adata_batch,
            gene_list=all_genes,
            library_size=LIBRARY_SIZE,
            return_numpy=True,
        )   # shape: (total_batch_cells, n_genes)

        # STEP 3: Vision scores for the entire batch at once
        scores_all, _ = vision_scores_from_expr(
            expr_all, gene_sets, gene_index, LOG_EPS
        )   # shape: (total_batch_cells, n_gene_sets)

        # STEP 4: Slice back to each group and compute differentials
        # First pass: collect DMSO means per group
        dmso_means = {}
        for g_idx, category, start, end, _ in batch_meta:
            if category == "dmso":
                dmso_means[g_idx] = scores_all[start:end, :].mean(axis=0)

        # Second pass: compute drug - DMSO per drug per group
        for g_idx, category, start, end, _ in batch_meta:
            if category == "dmso":
                continue
            gm      = group_meta[g_idx]
            d_idx   = gm["drug_names"].index(category)
            drug_mean = scores_all[start:end, :].mean(axis=0)
            gm["mc_diff"][s, d_idx, :] = drug_mean - dmso_means[g_idx]

    # ── Save results for each group ───────────────────────────────────────────
    print("\nSaving results...")
    for gm in group_meta:
        diff_scores  = np.median(gm["mc_diff"], axis=0)   # (n_comp, n_sets)
        labels       = [
            f"{drug} | {gm['plate']} | {gm['cell_line']}"
            for drug in gm["drug_names"]
        ]
        partial_path = PARTIAL_DIR / f"group_{gm['group_id']:04d}.parquet"
        pd.DataFrame(
            diff_scores,
            index=labels,
            columns=set_names,
            dtype=np.float32,
        ).to_parquet(partial_path)
        print(f"  Group {gm['group_id']} saved → {partial_path}")

    print(f"\nJob {job_id} complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_groups():
    print("Merging partial group files...")
    parts = sorted(PARTIAL_DIR.glob("group_*.parquet"))
    print(f"  Found {len(parts)} group files.")
    if not parts:
        print("No files found.")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], axis=0)
    df.to_parquet(OUTPUT_SCORES)
    print(f"Done! Final matrix shape: {df.shape}")
    print(f"Saved to: {OUTPUT_SCORES}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job_id", type=int,
                       help="Job index (from SLURM_ARRAY_TASK_ID)")
    group.add_argument("--merge", action="store_true",
                       help="Merge all partial outputs into final parquet")
    args = parser.parse_args()

    if args.merge:
        merge_groups()
    else:
        process_job(args.job_id)