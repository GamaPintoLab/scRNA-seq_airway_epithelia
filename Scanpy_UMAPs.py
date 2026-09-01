import scanpy as sc
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# 0) Project setup
# ----------------------------
project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq")

# input: HVG-annotated objects
in_dir = project_dir / "analysis" / "pearson_HVGs_per_sample"

# output: clustering + UMAP results
out_dir = project_dir / "analysis" / "leiden_per_sample"
out_dir.mkdir(parents=True, exist_ok=True)

sample_files = {
    "hNEC": in_dir / "hNEC_pearson_norm_HVGs.h5ad",
    "BCI_WT": in_dir / "BCI_WT_pearson_norm_HVGs.h5ad",
}

# Parameters
N_PCA_COMPS = 50      
N_NEIGHBORS = 15
N_PCS_NEIGHBORS = 15  
LEIDEN_RES = 1.0

# ----------------------------
# 1) Loop over samples
# ----------------------------
for sid, f in sample_files.items():
    print(f"\n=== PCA + elbow + neighbors + Leiden + UMAP for {sid} ===")
    adata = sc.read_h5ad(f)

    # Ensure the observation and variable names are unique
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # ----------------------------
    # 2) Checks: HVGs + Pearson residuals should exist
    # ----------------------------
    if "highly_variable" not in adata.var:
        raise KeyError(f"{sid}: adata.var['highly_variable'] not found. Run HVG selection first.")
    print(f"{sid}: HVGs flagged = {int(adata.var['highly_variable'].sum())}")

    if "pearson_residuals" not in adata.layers:
        raise KeyError(f"{sid}: adata.layers['pearson_residuals'] not found. Run Pearson residual normalization first.")

    # ----------------------------
    # 3) Compute PCA
    # ----------------------------
    try:
        sc.pp.pca(
            adata,
            n_comps=N_PCA_COMPS,
            layer="pearson_residuals",
            mask_var="highly_variable", 
            svd_solver="arpack",
            random_state=0,
        )
    except TypeError:
        sc.pp.pca(
            adata,
            n_comps=N_PCA_COMPS,
            layer="pearson_residuals",
            use_highly_variable=True,   
            svd_solver="arpack",
            random_state=0,
        )

    # ----------------------------
    # 4) Elbow plot
    # ----------------------------
    vr = np.asarray(adata.uns["pca"]["variance_ratio"][:N_PCA_COMPS], dtype=float)  
    x = np.arange(1, len(vr) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(x, vr, marker="o", linestyle="-", markersize=2.5, color="black")

    k = int(N_PCS_NEIGHBORS)
    if 1 <= k <= len(vr):
        plt.scatter([k], [vr[k - 1]], color="red", s=60, zorder=5)  

    plt.xlabel("PCs")
    plt.ylabel("Variance ratio")
    plt.xticks(np.arange(0, N_PCA_COMPS + 1, 10))
    #plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sid}_pca_elbow.png", dpi=300, bbox_inches="tight")
    plt.close()


    # ----------------------------
    # 5) Neighbors 
    # ----------------------------
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS_NEIGHBORS,
        use_rep="X_pca",
        metric="euclidean",
        random_state=0,
    )

    # ----------------------------
    # 6) Clustering (Leiden algorithm)
    # ----------------------------
    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RES,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    # ----------------------------
    # 6.5) Dendrogram (cluster distances)
    # ----------------------------
    sc.tl.dendrogram(
        adata,
        groupby="leiden",
        use_rep="X_pca",
        n_pcs=N_PCS_NEIGHBORS,
    )

    sc.pl.dendrogram(
        adata,
        groupby="leiden",
        show=False,
    )

    plt.savefig(out_dir / f"{sid}_leiden_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 7) UMAP + plot
    # ----------------------------
    sc.tl.umap(adata, random_state=0)

    sc.pl.umap(
        adata,
        color=["leiden"],
        legend_loc="on data",
        legend_fontsize=10,   
        frameon=False,
        show=False,
    )

    plt.savefig(out_dir / f"{sid}_umap_leiden_labeled.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 8) Save the updated object
    # ----------------------------
    out_file = out_dir / f"{sid}_pearson_leiden_umap.h5ad"
    print(f"Writing {sid} to {out_file}")
    adata.write(out_file)
