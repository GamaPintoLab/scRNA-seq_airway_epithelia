
import scanpy as sc
from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# 0) Project setup
# ----------------------------

project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq")  
in_dir      = project_dir / "analysis" / "scDblFinder_results"  
out_dir     = project_dir / "analysis" / "pearson_norm_per_sample"
out_dir.mkdir(parents=True, exist_ok=True)

sample_files = {
    "hNEC":   in_dir / "hNEC_singlets.h5ad",
    "BCI_WT": in_dir / "BCI_WT_singlets.h5ad",
}

for sid, f in sample_files.items():
    print(f"\n=== Processing {sid} ===")
    adata = sc.read_h5ad(f)

    # Make observation and variable names unique 
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # ----------------------------
    # 1) INPUT: define counts layer
    # ----------------------------
    if "counts" in adata.layers:
        X_counts = adata.layers["counts"]
    elif adata.raw is not None:
        X_counts = adata.raw.X
    else:
        X_counts = adata.X

    # store counts as CSR
    X_counts = sparse.csr_matrix(X_counts).copy()
    adata.layers["counts"] = X_counts

    # drop genes that are 0 in all cells
    gene_mask, _ = sc.pp.filter_genes(adata.layers["counts"], min_cells=1, inplace=False)
    n_genes_before = adata.n_vars
    n_genes_kept = int(gene_mask.sum())
    n_genes_removed = n_genes_before - n_genes_kept
    print(f"{sid}: filter_genes(min_cells=1) removed {n_genes_removed}/{n_genes_before} genes (kept {n_genes_kept})")

    adata = adata[:, gene_mask].copy()

    # ----------------------------
    # 2) OUTPUT: compute Pearson residuals into a new layer
    # ----------------------------
    res = sc.experimental.pp.normalize_pearson_residuals(
        adata,
        theta=100,
        clip=None,
        layer="counts",       
        check_values=True,
        inplace=False,         
    ) 
    adata.layers["pearson_residuals"] = res["X"]  

    # set working matrix
    adata.X = adata.layers["pearson_residuals"]


    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 1) Total counts per cell
    total_counts = np.array(adata.layers["counts"].sum(axis=1)).ravel()
    sns.histplot(total_counts, bins=100, kde=False, ax=axes[0])
    axes[0].set_title(f"{sid} – total counts")
    axes[0].set_xlabel("Total counts per cell")
    axes[0].set_ylabel("Number of cells")

    # 2) Sum of Pearson residuals per cell
    resid_sums = np.array(adata.layers["pearson_residuals"].sum(axis=1)).ravel()
    sns.histplot(resid_sums, bins=100, kde=False, ax=axes[1])
    axes[1].set_title(f"{sid} – Pearson residual sums")
    axes[1].set_xlabel("Sum of Pearson residuals per cell")
    axes[1].set_ylabel("Number of cells")

    plt.tight_layout()
    plt.savefig(out_dir / f"{sid}_hist_counts_vs_residuals.png", dpi=300)
    plt.close()


    # ----------------------------
    # 3) Save normalized objects
    # ----------------------------
    out_file = out_dir / f"{sid}_pearson_norm.h5ad"
    print(f"Writing {sid} to {out_file}")
    adata.write(out_file)
