# ############################################################################ #
# Integrate sections
# ############################################################################ #

# Integrate the different sections using Harmony.

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)

library(Seurat)
library(harmony)

## --------------------------------------------------------------------------- #
## Define input/output                                                    ----
## --------------------------------------------------------------------------- #

### Root paths
paths          <- list()
paths$data_inp <- here("data_input")
paths$data_pro <- here("data_processed")

### Specific paths
paths$data <- here(paths$data_pro, "020_seurat-pipeline")
paths$out  <- paths$data

### General input
general <- list(sections = here(paths$data_inp, "sections.csv"),
                seurat   = here(paths$data    , "initial-seurat.rds"))

### Output
outputs <- list(seurat = here(paths$out, "integrated-seurat.rds"))

## --------------------------------------------------------------------------- #
## Define parameters/optimizers                                           ----
## --------------------------------------------------------------------------- #

parameters <- list(
  hvgs   = 1000,
  pcs    = 10,
  theta  = c(4, 2, 1)  # for Sample, Section_ID, Experiment
)

# ============================================================================ #
# Prepare data                                                            ====
# ============================================================================ #

### Load sections details
sections <- read_csv(general$sections)

### Load seurat
seurat <- read_rds(general$seurat)

### Remove unwanted cells
seurat <- seurat |> 
  subset(Initial_celltype != "metaxylem")

### Remove undetected genes
genes_count    <- rowSums(seurat@assays$RNA@counts)
genes_detected <- names(genes_count[genes_count > 0])
seurat         <- seurat |> 
  subset(features = genes_detected)

rm(genes_count, genes_detected)

### Add Sample and Experiment variables
seurat@meta.data <- seurat@meta.data |> 
  as_tibble(rownames = "Cell_ID") |> 
  left_join(sections, by = "Section_ID") |> 
  relocate(Experiment, Sample, .after = Section_ID) |> 
  column_to_rownames("Cell_ID")

# ============================================================================ #
# Integrate                                                               ====
# ============================================================================ #

seurat <- seurat |> 
  # Normalize
  NormalizeData() |> 
  # Find Highly Variable Genes
  FindVariableFeatures(nfeatures = parameters[["hvgs"]]) |> 
  # Scale
  ScaleData() |> 
  # Perform PCA
  RunPCA(npcs = parameters[["pcs"]]) |> 
  # Integrate with harmony
  RunHarmony(group.by.vars = c("Sample", "Section_ID", "Experiment"),
             theta         = parameters[["theta"]],
             reduction     = "pca",
             assay.use     = "RNA",
             max.iter.harmony = 20)

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

write_rds(seurat, outputs$seurat, compress = "gz")
