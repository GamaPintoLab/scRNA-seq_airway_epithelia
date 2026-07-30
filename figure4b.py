import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import sparse


in_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq/analysis/joint_celltypist_internal_prescott")
adata = sc.read_h5ad(in_dir / "joint_prescott_celltypist_internal.h5ad")

analysis_dir = Path(r"C:/Users/arauj/Downloads/Research/scrnaseq/analysis")
out_dir = analysis_dir / "cftr_expression_by_celltype_prescott_merged_remake"
out_dir.mkdir(parents=True, exist_ok=True)

sample_col = "sample"
celltype_col = "celltypist_airway_majority"
gene = "CFTR"

sample_rename = {
    "HTO25": "primary",
    "HTO28": "primary",
    "HTO30": "primary",
    "HTO44": "bci",
    "HTO51": "bci",
    "HTO52": "bci",
}

if sample_col not in adata.obs.columns:
    raise KeyError(f"'{sample_col}' not found in adata.obs")
if celltype_col not in adata.obs.columns:
    raise KeyError(f"'{celltype_col}' not found in adata.obs")
if gene not in adata.var_names:
    raise KeyError(f"'{gene}' not found in adata.var_names")

adata.obs[sample_col] = adata.obs[sample_col].astype(str)
adata.obs[celltype_col] = adata.obs[celltype_col].astype(str)

adata.obs["merged_sample"] = adata.obs[sample_col].map(sample_rename)
adata = adata[adata.obs["merged_sample"].notna()].copy()

if adata.raw is not None and gene in adata.raw.var_names:
    X = adata.raw[:, [gene]].X
elif "counts" in adata.layers:
    X = adata[:, [gene]].layers["counts"]
else:
    X = adata[:, [gene]].X

if sparse.issparse(X):
    expr = X.toarray().ravel()
else:
    expr = np.asarray(X).ravel()

if np.nanmin(expr) < 0:
    raise ValueError("Selected expression matrix contains negative values. Use raw or log-normalized data, not scaled data.")

df = pd.DataFrame({
    "sample": adata.obs["merged_sample"].values,
    "celltype": adata.obs[celltype_col].values,
    "cftr": expr
})

sample_order = ["bci", "primary"]
preferred_celltype_order = [
    "Basal",
    "Dividing_Basal",
    "Suprabasal",
    "Deuterosomal",
    "Ciliated",
    "Secretory_Goblet",
    "Ionocyte_n_Brush",
    "Neuroendocrine",
    "NK_CD16hi",
]

present_celltypes = [ct for ct in preferred_celltype_order if ct in df["celltype"].unique()]

sum_expr = df.pivot_table(
    index="sample",
    columns="celltype",
    values="cftr",
    aggfunc="sum",
    fill_value=0
).reindex(index=sample_order, columns=present_celltypes)

sum_expr = sum_expr.fillna(0)
sum_expr = sum_expr.loc[sum_expr.sum(axis=1) > 0, :]
sum_expr = sum_expr.loc[:, sum_expr.sum(axis=0) > 0]

if sum_expr.shape[0] == 0 or sum_expr.shape[1] == 0:
    raise ValueError("No non-zero CFTR signal left to plot after filtering.")

percent_table = sum_expr.div(sum_expr.sum(axis=1), axis=0) * 100

sum_expr.to_csv(out_dir / "cftr_sum_expression_by_merged_sample_celltype.csv")
percent_table.to_csv(out_dir / "cftr_percent_contribution_by_merged_sample_celltype.csv")

sample_display = {
    "primary": "HBE",
    "bci": "BCi",
}

celltype_display = {
    "Basal": "Basal",
    "Dividing_Basal": "Dividing basal",
    "Suprabasal": "Suprabasal",
    "Deuterosomal": "Deuterosomal",
    "Ciliated": "Ciliated",
    "Secretory_Goblet": "Secretory goblet",
    "Ionocyte_n_Brush": "Ionocyte / tuft-like",
    "Neuroendocrine": "Neuroendocrine",
    "NK_CD16hi": "NK CD16hi",
}

celltype_palette = {
    "Basal": "#0d3b66",
    "Dividing_Basal": "#0099ff",
    "Suprabasal": "#7bb4d4",
    "Deuterosomal": "#0E800E",
    "Ciliated": "#93b438",
    "Secretory_Goblet": "#A78BC4",
    "Ionocyte_n_Brush": "#ff0000",
    "Neuroendocrine": "#7f7f7f",
    "NK_CD16hi": "#d57c29",
}

plot_table = percent_table.copy()
plot_table.index = [sample_display.get(x, x) for x in plot_table.index]
plot_table = plot_table.rename(columns=celltype_display)

plot_colors = {
    celltype_display.get(ct, ct): celltype_palette[ct]
    for ct in percent_table.columns
    if ct in celltype_palette
}

CM_TO_IN = 1 / 2.54
fig_width_in = 5 * CM_TO_IN
fig_height_in = 7.2 * CM_TO_IN

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
plot_table.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.5,
    legend=False,
    color=plot_colors
)

ax.set_ylabel("CFTR expression contribution (%)", fontsize=7)
ax.set_ylim(0, 100)
ax.tick_params(axis="y", labelsize=7)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("")
ax.set_xticklabels(plot_table.index, rotation=0, fontsize=8)

for i, sample in enumerate(plot_table.index):
    cumulative = 0
    for celltype in plot_table.columns:
        value = plot_table.loc[sample, celltype]
        if value > 4:
            ax.text(
                i,
                cumulative + value / 2,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold"
            )
        cumulative += value

plt.tight_layout()
plt.savefig(
    out_dir / "cftr_contribution_prescott_merged_matched_figure2b.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Saved outputs in: {out_dir}")