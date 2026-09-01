import scanpy as sc
import anndata as ad
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

# ----------------------------
# 0) Project and data setup
# ----------------------------

project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq/analysis")
data_dir = project_dir / "data"
results_dir = project_dir / "scanpy_QC"
results_dir.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = str(results_dir)

# Map sample IDs to the files
samples = {
    "hNEC":   data_dir / "hNEC"   / "cellbender_filtered.h5",
    "BCI_WT": data_dir / "BCi_WT" / "cellbender_filtered.h5",
}

adatas: dict[str, ad.AnnData] = {}

# ----------------------------
# 1) Load each sample
# ----------------------------

for sample_id, filepath in samples.items():
    print(f"Loading {sample_id} from {filepath}")
    adata = sc.read_10x_h5(str(filepath))
    adata.var_names_make_unique()
    adata.obs["sample"] = sample_id
    adatas[sample_id] = adata

# Print data dimensions
for sid, adata in adatas.items():
    print(f"{sid}: {adata.shape}")

# ----------------------------
# 2) QC metrics per sample
# ----------------------------

for sid, adata in adatas.items():
    print(f"Calculating QC metrics for {sid}")
    # Annotate mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))

    # Calculate per-cell and per-gene QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=True,
        inplace=True,
    )

# ----------------------------
# 3) QC thresholds
# ----------------------------

min_genes = 1750
max_pct_mt = 20.0
min_reads = 5000
zoom_total_max = 30000
for sid, adata in adatas.items():
    # Boolean QC flag for each cell
    adata.obs["high_quality"] = (
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["total_counts"]      >= min_reads) &
        (adata.obs["pct_counts_mt"]     <= max_pct_mt)
    )

# ----------------------------
# 4) QC plots per sample
# ----------------------------

for sid, adata in adatas.items():
    print(f"QC plots for {sid}")

    # Zooming settings
    mask_zoom = adata.obs["total_counts"] < zoom_total_max

    # 4.1) Violin plots: n_genes, total_counts, mitochondrial
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    fig = plt.gcf()
    axes = fig.axes

    ax_genes = axes[0]
    ax_genes.axhline(min_genes, color="red", linestyle="--", linewidth=1)

    ax_mt = axes[2]
    ax_mt.axhline(max_pct_mt, color="red", linestyle="--", linewidth=1)

    fig.set_size_inches(9, 4)
    fig.savefig(
        results_dir / f"{sid}_qc_violin.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4.2) Scatter plot: total_counts vs mitochondrial (zoomed)
    sc.pl.scatter(
        adata[mask_zoom],
        x="total_counts",
        y="pct_counts_mt",
        show=False,
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.axhline(max_pct_mt, color="red", linestyle="--", linewidth=1)
    ax.axvline(min_reads,  color="red", linestyle="--", linewidth=1)
    fig.set_size_inches(5, 4)
    fig.savefig(
        results_dir / f"{sid}_total_vs_mito_zoom.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4.3) Scatter plot: total_counts vs mitochondrial
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    fig.savefig(
        results_dir / f"{sid}_total_vs_mito.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4.4) Scatter plot: total_counts vs n_genes_by_counts (zoomed)
    sc.pl.scatter(
        adata[mask_zoom],
        x="total_counts",
        y="n_genes_by_counts",
        show=False,
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.axhline(min_genes, color="red", linestyle="--", linewidth=1)
    ax.axvline(min_reads,  color="red", linestyle="--", linewidth=1)
    fig.set_size_inches(5, 4)
    fig.savefig(
        results_dir / f"{sid}_total_vs_genes_zoom.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4.5) Scatter plot: total_counts vs n_genes_by_counts
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    fig.savefig(
        results_dir / f"{sid}_total_vs_genes.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

# ----------------------------
# 5) QC UMAP per sample 
# ----------------------------
for sid, adata in adatas.items():
    print(f"QC UMAP for {sid}")


    adata_umap = adata.copy()

    # Light preprocessing for UMAP settings
    sc.pp.normalize_total(adata_umap, target_sum=1e4)
    sc.pp.log1p(adata_umap)
    sc.pp.pca(adata_umap)
    sc.pp.neighbors(adata_umap)
    sc.tl.umap(adata_umap)

    # UMAP colored by mitochondrial percentage
    sc.pl.umap(
        adata_umap,
        color=["pct_counts_mt"],
        cmap="viridis",
        show=False,
    )
    plt.savefig(
        results_dir / f"{sid}_UMAP_pct_mt_all_cells.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # UMAP colored by QC flag
    sc.pl.umap(
        adata_umap,
        color=["high_quality"],
        show=False,
    )
    plt.savefig(
        results_dir / f"{sid}_UMAP_high_quality_flag.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ----------------------------
# 6) Filter and write 10x matrices
# ----------------------------

for sid, adata in adatas.items():
    print(f"Filtering {sid}")
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["total_counts"]      >= min_reads) &
        (adata.obs["pct_counts_mt"]     <= max_pct_mt)
    )
    print(f"{sid}: keeping {cell_mask.sum()} / {adata.n_obs} cells")
    adatas[sid] = adata[cell_mask].copy()

def write_10X_h5(adata, file):
    """Write AnnData to a 10X Genomics formatted HDF5 file compatible with Seurat::Read10X_h5."""
    file = str(file)
    if not file.endswith(".h5"):
        file = file + ".h5"
    
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)
    X = X.T
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    X = csc_matrix(X.T) 
    def str_max(x):
        return max(len(str(i)) for i in x) + 1

    with h5py.File(file, "w") as w:
        grp = w.create_group("matrix")
        grp.create_dataset("data",    data=X.data.astype(np.float32))
        grp.create_dataset("indices", data=X.indices.astype(np.int32))
        grp.create_dataset("indptr",  data=X.indptr.astype(np.int32))
        grp.create_dataset("shape",   data=np.array(X.shape, dtype=np.int32))

        # barcodes
        barcodes = np.array(adata.obs_names, dtype=f"S{str_max(adata.obs_names)}")
        grp.create_dataset("barcodes", data=barcodes)

        # features
        ftrs = grp.create_group("features")
        gene_names = np.array(adata.var_names, dtype=f"S{str_max(adata.var_names)}")
        ftrs.create_dataset("id",   data=gene_names)
        ftrs.create_dataset("name", data=gene_names)

        # feature_type
        if "feature_type" in adata.var.columns:
            ft = np.array(adata.var["feature_type"], dtype=f"S{str_max(adata.var['feature_type'])}")
        else:
            ft = np.array(["Gene Expression"] * adata.n_vars, dtype=f"S{len('Gene Expression')+1}")
        ftrs.create_dataset("feature_type", data=ft)



for sid, adata in adatas.items():
    out_file = results_dir / f"{sid}_filtered_10x.h5"
    print(f"Writing 10x HDF5 for {sid} to {out_file}")
    write_10X_h5(adata, out_file)
 
    

