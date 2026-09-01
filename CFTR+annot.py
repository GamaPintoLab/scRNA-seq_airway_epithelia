import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

projectdir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq/analysis")
indir = projectdir / "joint_celltypist_internal"
outdir = projectdir / "cftr_positive_h5ad"
outdir.mkdir(parents=True, exist_ok=True)

infile = indir / "joint_hNEC_BCIWT_celltypist_internal.h5ad"
outfile = outdir / "joint_hNEC_BCIWT_celltypist_internal_cftr_positive.h5ad"

gene = "CFTR"
obs_col_bool = "cftr_positive_pr_gt0"
obs_col_label = "cftr_positive_pr_gt0_label"
threshold = 0.0

adata = sc.read_h5ad(infile)

if gene not in adata.var_names:
    raise KeyError(f"{gene} not found in adata.var_names")

X = adata[:, [gene]].X
if sparse.issparse(X):
    vals = np.asarray(X.toarray()).ravel()
else:
    vals = np.asarray(X).ravel()

vals = vals.astype(float)
if not np.isfinite(vals).all():
    raise ValueError("CFTR values contain non-finite entries.")

adata.obs[obs_col_bool] = vals > threshold
adata.obs[obs_col_label] = pd.Categorical(
    np.where(adata.obs[obs_col_bool], "CFTR_positive", "CFTR_negative"),
    categories=["CFTR_negative", "CFTR_positive"]
)

adata.uns["cftr_positive_definition"] = {
    "gene": gene,
    "source": "adata.X",
    "value_type": "Pearson residual",
    "rule": f"{gene} > {threshold}",
    "bool_obs_column": obs_col_bool,
    "label_obs_column": obs_col_label,
}

print(f"Input file: {infile}")
print(f"Output directory: {outdir}")
print(f"Output file: {outfile}")
print(f"Total cells: {adata.n_obs:,}")
print(f"CFTR-positive cells: {int(adata.obs[obs_col_bool].sum()):,}")
print(f"CFTR-negative cells: {int((~adata.obs[obs_col_bool]).sum()):,}")

print("\nCounts:")
print(adata.obs[obs_col_label].value_counts())

adata.write_h5ad(outfile)
print(f"\nSaved updated h5ad to: {outfile}")