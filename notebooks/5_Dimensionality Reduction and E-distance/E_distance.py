"""
E-distance Computation (plate-by-plate)
========================================
Computes E-distance for each (drug, cell_line) comparison within a plate,
using the scVI 10-dimensional latent representation.

E-distance formula (Peidli et al. 2024):
  E = (2 * mean_cross - mean_treat - mean_ctrl) * adjustment
  where:
    mean_cross = mean pairwise distance between treated and control cells
    mean_treat = mean pairwise distance within treated cells
    mean_ctrl  = mean pairwise distance within control cells
    adjustment = (n_treat * n_ctrl) / (n_treat + n_ctrl)


Output per plate:
  e_distance_plate{N}.parquet
  Columns: drug, plate, cell_line, e_distance, n_treated, n_control
"""

import argparse
import numpy as np
import pandas as pd
import torch
import anndata
import scvi.hub
from pathlib import Path
from scipy.spatial.distance import cdist
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
OUTPUT_DIR = BASE_DIR / "Data" / "e_distance"

DMSO_LABEL = "DMSO_TF"
CELL_LINE_COL = "Cell_Name_Vevo"   # column for cell line labels
DRUG_COL = "drug"
PLATE_COL = "plate"
MIN_CELLS = 50          # discard comparisons with < 50 cells in either group
MAX_CELLS = 200         # subsample larger groups for speed
N_LATENT = 10           # scVI latent dimensionality
BATCH_SIZE = 5000       # for get_latent_representation

# Plate mapping: array index → plate name
PLATES = [f"plate{i}" for i in range(1, 15)]  # plate1 through plate14

# ─────────────────────────────────────────────
# PARSE ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--plate_idx", type=int, required=True,
                    help="Index into PLATES list (0-13)")
args = parser.parse_args()

plate_name = PLATES[args.plate_idx]
print("=" * 60)
print(f"E-DISTANCE COMPUTATION: {plate_name}")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD MODEL AND DATA
# ─────────────────────────────────────────────
t0 = time.time()
print(f"\n1. Loading Tahoe SCVI model...")

tahoe_hubmodel = scvi.hub.HubModel.pull_from_huggingface_hub(
    repo_name="vevotx/Tahoe-100M-SCVI-v1",
    cache_dir=CACHE_DIR,
)
tahoe = tahoe_hubmodel.model

device = "cuda" if torch.cuda.is_available() else "cpu"
tahoe.to_device(device)
print(f"   Model on: {device}")

print(f"2. Loading adata from: {DATA_PATH}")
adata = anndata.read_h5ad(DATA_PATH)
print(f"   Full adata: {adata.n_obs} cells × {adata.n_vars} genes")

# ─────────────────────────────────────────────
# 2. SUBSET TO PLATE
# ─────────────────────────────────────────────
print(f"\n3. Subsetting to {plate_name}...")

# Filter to full-pass cells on this plate
plate_mask = adata.obs[PLATE_COL] == plate_name
if "pass_filter" in adata.obs.columns:
    plate_mask = plate_mask & (adata.obs["pass_filter"] == "full")

plate_adata = adata[plate_mask].copy()
n_plate = plate_adata.n_obs
print(f"   Cells in {plate_name}: {n_plate}")

if n_plate == 0:
    print(f"   No cells found for {plate_name}. Exiting.")
    exit(0)

# ─────────────────────────────────────────────
# 3. GET LATENT REPRESENTATIONS
# ─────────────────────────────────────────────
print(f"\n4. Computing {N_LATENT}D latent representations...")
t1 = time.time()

latent = tahoe.get_latent_representation(
    plate_adata,
    batch_size=BATCH_SIZE,
)
# latent is (n_cells, 10) numpy array
print(f"   Latent shape: {latent.shape}")
print(f"   Took {time.time() - t1:.1f}s")

# Attach to obs for easy grouping
plate_obs = plate_adata.obs.copy()
plate_obs["latent_idx"] = np.arange(len(plate_obs))

# ─────────────────────────────────────────────
# 4. COMPUTE E-DISTANCE PER COMPARISON
# ─────────────────────────────────────────────
print(f"\n5. Computing E-distances...")
t2 = time.time()

# Get DMSO control cells per cell_line
ctrl_mask = plate_obs[DRUG_COL] == DMSO_LABEL
ctrl_obs = plate_obs[ctrl_mask]

# Get treated cells (everything except DMSO)
treat_obs = plate_obs[~ctrl_mask]

# Unique cell lines in this plate
cell_lines = treat_obs[CELL_LINE_COL].unique()

results = []
skipped = 0

for cl in cell_lines:
    # Control cells for this cell line
    cl_ctrl = ctrl_obs[ctrl_obs[CELL_LINE_COL] == cl]
    ctrl_indices = cl_ctrl["latent_idx"].values
    n_ctrl = len(ctrl_indices)

    if n_ctrl < MIN_CELLS:
        continue

    # Subsample if too many control cells
    if n_ctrl > MAX_CELLS:
        ctrl_indices = np.random.choice(ctrl_indices, MAX_CELLS, replace=False)
        n_ctrl = MAX_CELLS
    ctrl_latent = latent[ctrl_indices]

    # Drugs tested on this cell line
    cl_treat = treat_obs[treat_obs[CELL_LINE_COL] == cl]
    drugs = cl_treat[DRUG_COL].unique()

    for drug in drugs:
        drug_cells = cl_treat[cl_treat[DRUG_COL] == drug]
        treat_indices = drug_cells["latent_idx"].values
        n_treat = len(treat_indices)

        if n_treat < MIN_CELLS:
            skipped += 1
            continue

        # Subsample if too many treated cells
        if n_treat > MAX_CELLS:
            treat_indices = np.random.choice(treat_indices, MAX_CELLS, replace=False)
            n_treat = MAX_CELLS
        treat_latent = latent[treat_indices]

        # Compute pairwise distances
        # Cross distances: treated vs control
        d_cross = cdist(treat_latent, ctrl_latent, metric="euclidean")
        mean_cross = d_cross.mean()

        # Within treated
        d_treat = cdist(treat_latent, treat_latent, metric="euclidean")
        mean_treat = d_treat[np.triu_indices(n_treat, k=1)].mean() if n_treat > 1 else 0.0

        # Within control
        d_ctrl = cdist(ctrl_latent, ctrl_latent, metric="euclidean")
        mean_ctrl = d_ctrl[np.triu_indices(n_ctrl, k=1)].mean() if n_ctrl > 1 else 0.0

        # E-distance with cell count adjustment
        e_dist = (2 * mean_cross - mean_treat - mean_ctrl) * (n_treat * n_ctrl) / (n_treat + n_ctrl)

        # Extract plate number for index compatibility with Vision scores
        plate_num = plate_name.replace("plate", "")

        results.append({
            "drug": drug,
            "plate": plate_num,
            "cell_line": cl,
            "e_distance": e_dist,
            "log1p_e_distance": np.log1p(e_dist),
            "n_treated": n_treat,
            "n_control": n_ctrl,
        })

print(f"   Computed {len(results)} E-distances")
print(f"   Skipped {skipped} comparisons (< {MIN_CELLS} cells)")
print(f"   Took {time.time() - t2:.1f}s")

# ─────────────────────────────────────────────
# 5. SAVE RESULTS
# ─────────────────────────────────────────────
if len(results) > 0:
    df = pd.DataFrame(results)

    # Create index matching Vision score format: 'drug | plate | cell_line'
    df["comparison_id"] = df["drug"] + " | " + df["plate"] + " | " + df["cell_line"]

    out_path = OUTPUT_DIR / f"e_distance_{plate_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n   Saved → {out_path}")
    print(f"   Shape: {df.shape}")
    print(f"\n   E-distance stats:")
    print(f"     mean: {df['e_distance'].mean():.4f}")
    print(f"     median: {df['e_distance'].median():.4f}")
    print(f"     max: {df['e_distance'].max():.4f}")
else:
    print("\n   WARNING: No valid comparisons found!")

total = time.time() - t0
print(f"\n{'=' * 60}")
print(f"DONE: {plate_name} in {total:.0f}s ({total/60:.1f} min)")
print(f"{'=' * 60}")