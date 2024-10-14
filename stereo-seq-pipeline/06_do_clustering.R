# ############################################################################ #
# Do clustering
# ############################################################################ #

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)

# Setup the python interpreter to "napari-sparrow" for using leidenalg
library(reticulate)
use_condaenv("napari-sparrow", required = TRUE)
py_module_available("leidenalg")  # Required to work

library(Seurat)

## --------------------------------------------------------------------------- #
## Define input/output                                                    ----
## --------------------------------------------------------------------------- #

### Root paths
paths          <- list()
paths$data_inp <- here("data_input")
paths$data_pro <- here("data_processed")

### Specific paths
paths$data  <- here(paths$data_pro, "020_seurat-pipeline")

### Inputs
seurat_in = here(paths$data, "integrated-seurat.rds")

### Outputs
seurat_out = here(paths$data, "clustered-seurat.rds")

## --------------------------------------------------------------------------- #
## Define parameters                                                      ----
## --------------------------------------------------------------------------- #

parameters <- list(
  reduction = "harmony",
  dims      = 10       ,
  algorithm = 4        ,
  k         = 20       ,
  res       = 0.24
)
parameters[["n.neighbors"]] = parameters[["k"]]

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Load seurat
seurat <- read_rds(seurat_in)

## Find neighbors
seurat <- seurat %>%
    FindNeighbors(reduction  = parameters[["reduction"]],
                  dims       = 1:parameters[["dims"]],
                  k.param    = parameters[["k"]])

## Find clusters
seurat <- seurat %>% 
    FindClusters(algorithm  = parameters[["algorithm"]],
                 resolution = parameters[["res"]])

## Tidy metadata
seurat <- seurat |> 
  AddMetaData(seurat@meta.data[["seurat_clusters"]], "Cluster")
seurat[["seurat_clusters"]]                         <- NULL
seurat[[glue("RNA_snn_res.{parameters[['res']]}")]] <- NULL

## Perform UMAP
seurat <- seurat |> 
  RunUMAP(reduction   = parameters[["reduction"]],
          dims        = 1:parameters[["dims"]],
          n.neighbors = parameters[["n.neighbors"]],  # default = 30L,
          min.dist    = 0.3)                          # default = 0.3

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

saveRDS(seurat, seurat_out)
