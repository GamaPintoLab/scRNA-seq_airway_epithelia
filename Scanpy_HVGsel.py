import scanpy as sc
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# 0) Project setup
# ----------------------------
project_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq")
in_dir = project_dir / "analysis" / "pearson_norm_per_sample"
out_dir = project_dir / "analysis" / "pearson_HVGs_per_sample"
out_dir.mkdir(parents=True, exist_ok=True)

sample_files = {
    "hNEC": in_dir / "hNEC_pearson_norm.h5ad",
    "BCI_WT": in_dir / "BCI_WT_pearson_norm.h5ad",
}

N_HVG = 2000 

# ----------------------------
# 1) Loop over samples
# ----------------------------
for sid, f in sample_files.items():
    print(f"\n=== HVG selection (Pearson residuals) for {sid} ===")
    adata = sc.read_h5ad(f)

    # Make names unique
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # ----------------------------
    # 2) Highly variable gene selection
    # ----------------------------

    sc.experimental.pp.highly_variable_genes(
        adata,
        flavor="pearson_residuals",
        layer="counts",
        n_top_genes=N_HVG,   
        inplace=True,
    ) 

    # ----------------------------
    # 3) Plot mean vs residual variance
    # ----------------------------
    means = adata.var["means"].to_numpy()
    res_var = adata.var["residual_variances"].to_numpy()

    # Plot only finite, positive points 
    valid = np.isfinite(means) & np.isfinite(res_var) & (means > 0) & (res_var > 0)
    x = means[valid]
    y = res_var[valid]

    logx = np.log10(x)
    logy = np.log10(y)

    bins = 100
    H, xedges, yedges = np.histogram2d(logx, logy, bins=bins)

    xidx = np.clip(np.digitize(logx, xedges) - 1, 0, H.shape[0] - 1)
    yidx = np.clip(np.digitize(logy, yedges) - 1, 0, H.shape[1] - 1)
    density = H[xidx, yidx]

    log_density = np.log10(density + 1)
    order = np.argsort(log_density)

    x_sorted = x[order]
    y_sorted = y[order]
    log_density_sorted = log_density[order]

    plt.figure(figsize=(6, 5))
    sc_plt = plt.scatter(
        x_sorted,
        y_sorted,
        c=log_density_sorted,
        s=5,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc_plt)
    cbar.set_label("log10(point density + 1)")

    # ---- cutoff  ----
    finite = np.isfinite(res_var)
    res_var_finite = res_var[finite]

    idx_sort = np.argsort(res_var_finite)[::-1]
    k = min(N_HVG - 1, len(res_var_finite) - 1)
    cutoff = res_var_finite[idx_sort][k]

    plt.axhline(
        cutoff,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{N_HVG}th gene residual variance",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean expression per gene (log scale)")
    plt.ylabel("Residual variance per gene (log scale)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{sid}_HVG_residual_variance_density_log.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------
    # 4) Write HVG list (txt format)
    # ----------------------------
    hvgs = adata.var_names[adata.var["highly_variable"]].to_list()
    hvgs_txt = out_dir / f"{sid}_HVGs.txt"
    with open(hvgs_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hvgs) + ("\n" if len(hvgs) else ""))
    print(f"{sid}: wrote {len(hvgs)} HVGs to {hvgs_txt}")

    # ----------------------------
    # 5) Save the updated object (with HVG flags)
    # ----------------------------
    out_file = out_dir / f"{sid}_pearson_norm_HVGs.h5ad"
    print(f"Writing {sid} HVG annotations to {out_file}")
    adata.write(out_file)
