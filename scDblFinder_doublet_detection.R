

library(Seurat)
library(SingleCellExperiment)
library(scDblFinder)
library(anndataR)
library(ggplot2)

tenx_root   <- "C:/Users/arauj/Downloads/Research/scrnaseq/analysis/scanpy_QC"
results_dir <- "C:/Users/arauj/Downloads/Research/scrnaseq/analysis/scDblFinder_results"
dir.create(results_dir, showWarnings = FALSE)

samples <- c("hNEC", "BCI_WT")

for (sid in samples) {
  cat("Processing", sid, "\n")
  
  h5_file <- file.path(tenx_root, paste0(sid, "_filtered_10x.h5"))
  
  counts <- Read10X_h5(h5_file)
  seu    <- CreateSeuratObject(counts = counts, project = sid)
  
  sce <- as.SingleCellExperiment(seu)
  sce$sample <- sid
  
  set.seed(123)
  sce <- scDblFinder(sce)
  
  
  
  # Scatter plot: total_counts vs n_genes
  df <- data.frame(
    n_genes      = sce$nFeature_RNA,
    total_counts = sce$nCount_RNA,
    class        = sce$scDblFinder.class
  )
  
  p <- ggplot(df, aes(x = total_counts, y = n_genes, color = class)) +
    geom_point(size = 0.2, alpha = 0.4) +
    scale_x_continuous(
      breaks = seq(0, max(df$total_counts, na.rm = TRUE), by = 100000)
    ) +
    theme_classic() +
    ggtitle(paste("QC with doublet calls -", sid))
  
  ggsave(
    filename = file.path(results_dir, paste0(sid, "_doublet_QC_scatter.png")),
    plot     = p,
    width    = 5,
    height   = 4,
    dpi      = 300
  )
  
  # Write full object with doublet labels ----
  out_h5ad_all <- file.path(results_dir, paste0(sid, "_with_doublet_calls.h5ad"))
  write_h5ad(sce, out_h5ad_all)
  
  
  singlet_mask <- sce$scDblFinder.class == "singlet"
  sce_singlet  <- sce[, singlet_mask]
  cat(sprintf("%s: %d → %d singlets (%.1f%% kept)\n", 
              sid, ncol(sce), ncol(sce_singlet), 
              ncol(sce_singlet)/ncol(sce)*100))
  
  # Subset to singlets and write singlet-only h5ad file
  singlet_mask <- sce$scDblFinder.class == "singlet"
  sce_singlet  <- sce[, singlet_mask]
  
  out_h5ad_singlets <- file.path(results_dir, paste0(sid, "_singlets.h5ad"))
  write_h5ad(sce_singlet, out_h5ad_singlets)
}
