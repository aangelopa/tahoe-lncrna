import torch
from pathlib import Path
import scvi.hub
import anndata
import numpy as np
import pandas as pd


# CONFIGURATION

BASE_DIR = Path.home() / "Thesis"
DATA_DIR = BASE_DIR / "Data"
CACHE_DIR = Path("/home/a/aangelopa/Thesis/Data/tahoe_cache")
SNAPSHOT_PATH = Path("/home/a/aangelopa/Thesis/Data/tahoe_cache/models--vevotx--Tahoe-100M-SCVI-v1/snapshots/b5283a73fbbed812a95264ace360da538b20af89")
DATA_PATH = SNAPSHOT_PATH / "adata.h5ad"
GENE_LIST_PATH = DATA_DIR / "all_unique_genes.txt"  
OUTPUT_TXT = "scvi_HVGs.txt"

# lOADING MODEL


print("1. Loading Tahoe...")

tahoe_hubmodel = scvi.hub.HubModel.pull_from_huggingface_hub(
    repo_name="vevotx/Tahoe-100M-SCVI-v1",
    cache_dir=CACHE_DIR,
)

tahoe = tahoe_hubmodel.model

tahoe

tahoe.adata

device = "cuda" if torch.cuda.is_available() else "cpu"
tahoe.to_device(device)
print(f"Model loaded on: {device}")


print("SUCCESS: Model is ready.")

# Calculating HVGs

print(f"2. Loading data from: {DATA_PATH}")
adata = anndata.read_h5ad(DATA_PATH)

print("-" * 30)
print(f"3. Loading the genes from {GENE_LIST_PATH}...")

with open(GENE_LIST_PATH, 'r') as f:
    my_genes = [line.strip() for line in f if line.strip()]


# Filter genes that exist in the model
valid_genes = [g for g in my_genes if g in adata.var_names]
print(f"-> Found {len(valid_genes)} valid genes out of {len(my_genes)} in the list.")



print("4. Sampling 10,000 random cells...")
random_indices = np.random.choice(adata.n_obs, 10000, replace=False)

# Create the subset
subset = adata[random_indices].copy()

print("5. Generating model-denoised expression...")
denoised_expression = tahoe.get_normalized_expression(
    subset,
    gene_list=valid_genes,
    library_size=10e4
)

print("6. Calculating standard deviations...")
# Calculate standard deviation for each gene
gene_std = denoised_expression.std(axis=0)

# Create a DataFrame
results = pd.DataFrame({
    'gene': valid_genes,
    'std_dev': gene_std.values
})

# Sort by variability (highest first)
results = results.sort_values('std_dev', ascending=False)

# Take the top 50% as "Highly Variable"
median_std = results['std_dev'].median()
hvg_df = results[results['std_dev'] > median_std]

print("-" * 30)
print(f"Identified {len(hvg_df)} Highly Variable Genes (Top 50%).")
print("Top 5 most variable:")
print(hvg_df.head(5))


print(f"7. Saving to {OUTPUT_TXT}...")
with open(OUTPUT_TXT, 'w') as f:
    for gene in hvg_df['gene']:
        f.write(f"{gene}\n")

print("DONE! File created successfully!.")

