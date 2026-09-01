
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq")
in_dir = project_dir / "analysis" / "scDblFinder_results"

for sid in ["hNEC", "BCI_WT"]:
    print(f"Processing {sid}")
    adata = sc.read_h5ad(in_dir / f"{sid}_with_doublet_calls.h5ad")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # If counts are in a layer, move them to X
    print("Layers:", adata.layers.keys())
    if "counts" in adata.layers.keys():
        adata.X = adata.layers["counts"]

    # Simple preprocessing just for the UMAP
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color=["scDblFinder.class"],
        size=10,
        show=False,
    )
    plt.savefig(
        in_dir / f"{sid}_UMAP_scDblFinder_class.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()