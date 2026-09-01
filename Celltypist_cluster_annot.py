import scanpy as sc
import celltypist
from celltypist import models
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# 0) Project and I/O setup
# ----------------------------
project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq")

in_dir = project_dir / "analysis" / "leiden_per_sample"
out_dir = project_dir / "analysis" / "celltypist_per_sample"
out_dir.mkdir(parents=True, exist_ok=True)

sample_files = {
    "hNEC":   in_dir / "hNEC_pearson_leiden_umap.h5ad",
    "BCI_WT": in_dir / "BCI_WT_pearson_leiden_umap.h5ad",
}

# ----------------------------
# 1) CellTypist models
# ----------------------------
print("\n=== Available CellTypist models (short description) ===")
print(models.models_description())

chosen_models = [
    "Human_Lung_Atlas.pkl",
    "Cells_Lung_Airway.pkl",
]
print(f"\nUsing CellTypist models: {chosen_models}")
models.download_models(model=chosen_models)

# Majority voting aligned to Leiden clusters
OVER_CLUSTERING_KEY = "leiden"
MIN_PROP = 0.5

# Label style
LABEL_FONTSIZE = 7
LABEL_BBOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65)

# ----------------------------
# 2) Loop over samples
# ----------------------------
for sid, f in sample_files.items():
    print(f"\n=== CellTypist annotation for {sid} ===")
    adata = sc.read_h5ad(f)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if "X_umap" not in adata.obsm:
        raise KeyError(f"{sid}: missing adata.obsm['X_umap'] (run UMAP first).")
    if OVER_CLUSTERING_KEY not in adata.obs:
        raise KeyError(f"{sid}: missing adata.obs['{OVER_CLUSTERING_KEY}'] (run Leiden first).")

    if not pd.api.types.is_categorical_dtype(adata.obs[OVER_CLUSTERING_KEY]):
        adata.obs[OVER_CLUSTERING_KEY] = adata.obs[OVER_CLUSTERING_KEY].astype("category")

    # Loop over models
    for model_name in chosen_models:
        model_stem = Path(model_name).stem
        print(f"\n--- Annotating {sid} with model: {model_name} ---")

        # 2a) Build CellTypist input
        adata_ct = adata.copy()
        if "counts" in adata_ct.layers:
            adata_ct.X = adata_ct.layers["counts"]
        elif adata_ct.raw is not None:
            adata_ct.X = adata_ct.raw.X

        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)

        # 2b) CellTypist prediction
        predictions = celltypist.annotate(
            adata_ct,
            model=model_name,
            majority_voting=True,
            over_clustering=OVER_CLUSTERING_KEY,
            min_prop=MIN_PROP,
        )
        pl = predictions.predicted_labels

        maj_col = f"celltypist_{model_stem}_majority"
        adata.obs[maj_col] = pl["majority_voting"].astype("category")
        adata.uns[f"celltypist_{model_stem}_predicted_labels_table"] = pl

        # ----------------------------
        #Save cell counts per CellTypist label (per sample, per model)
        # ----------------------------
        counts = (
            adata.obs[maj_col]
            .value_counts(dropna=False)  # keep NaN/Unknown if present
            .rename_axis("cell_type")
            .reset_index(name="n_cells")
        )

        counts.to_csv(
            out_dir / f"{sid}_cellcounts_{model_stem}_majority.csv",
            index=False,
        )

        # 2c) UMAP coords as a DataFrame for computing medians
        umap = adata.obsm["X_umap"]
        tmp = pd.DataFrame(
            {
                "umap1": umap[:, 0],
                "umap2": umap[:, 1],
                "leiden": adata.obs[OVER_CLUSTERING_KEY].astype(str).values,
                "ct_majority": adata.obs[maj_col].astype(str).values,
            },
            index=adata.obs_names,
        )

        # ----------------------------
        # Plot A: Leiden colors + CellTypist labels per Leiden cluster
        # ----------------------------
        cluster_to_ct = (
            adata.obs.groupby(OVER_CLUSTERING_KEY)[maj_col]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )

        leiden_centers = tmp.groupby("leiden")[["umap1", "umap2"]].median()

        ax = sc.pl.umap(
            adata,
            color=OVER_CLUSTERING_KEY,
            legend_loc="none", 
            frameon=False,
            show=False,       
        )

        for cl, row in leiden_centers.iterrows():
            ax.text(
                row["umap1"], row["umap2"],
                cluster_to_ct.get(cl, "Unknown"),
                fontsize=LABEL_FONTSIZE,
                ha="center", va="center",
                bbox=LABEL_BBOX,
            )

        plt.savefig(
            out_dir / f"{sid}_umap_{model_stem}_leiden_colored_celltypist_labels.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ----------------------------
        # Plot B: CellTypist “clustering” (majority labels)
        # ----------------------------
        ct_centers = tmp.groupby("ct_majority")[["umap1", "umap2"]].median()

        ax = sc.pl.umap(
            adata,
            color=maj_col,
            legend_loc="none",  
            frameon=False,
            show=False,       
        )

        # Use category order for stable labeling 
        ct_order = list(adata.obs[maj_col].cat.categories) if pd.api.types.is_categorical_dtype(adata.obs[maj_col]) else list(ct_centers.index)

        for lab in ct_order:
            if lab not in ct_centers.index:
                continue
            row = ct_centers.loc[lab]
            ax.text(
                row["umap1"], row["umap2"],
                lab,
                fontsize=LABEL_FONTSIZE,
                ha="center", va="center",
                bbox=LABEL_BBOX,
            )

        plt.savefig(
            out_dir / f"{sid}_umap_{model_stem}_celltypist_colored_celltypist_labels.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save annotated object 
    out_file = out_dir / f"{sid}_pearson_leiden_umap_celltypist.h5ad"
    print(f"Writing {sid} with CellTypist labels to {out_file}")
    adata.write(out_file)
