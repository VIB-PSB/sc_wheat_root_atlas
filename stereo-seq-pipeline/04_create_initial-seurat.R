# ############################################################################ #
# Create the initial seurat
# ############################################################################ #

# Create the initial seurat from the count matrix. Merge all sections together.

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)

library(Seurat)

## --------------------------------------------------------------------------- #
## Define input/output                                                    ----
## --------------------------------------------------------------------------- #

### Root paths
paths          <- list()
paths$data_inp <- here("data_input")
paths$data_pro <- here("data_processed")

### Specific paths
paths$count_data <- here(paths$data_pro, "010_sparrow-pipeline/count-data")
paths$out        <- here(paths$data_pro, "020_seurat-pipeline")

### General input
general <- list(sections = here(paths$data_inp, "sections.csv"))

# Section-specific details
sections <- general$sections |> 
  read_csv(show_col_types = FALSE) |>  
  # Input
  mutate(
    Count_matrix      = glue("{paths$count_data}/{Section_ID}_count-data_matrix.mtx.gz"),
    Count_cells       = glue("{paths$count_data}/{Section_ID}_count-data_cells.csv"),
    Count_genes       = glue("{paths$count_data}/{Section_ID}_count-data_genes.csv"),
    Initial_celltypes = glue("{paths$count_data}/{Section_ID}_initial-celltypes.csv")
  )

### Output
outputs <- list(seurat = here(paths$out, "initial-seurat.rds"))

## --------------------------------------------------------------------------- #
## Define parameters                                                      ----
## --------------------------------------------------------------------------- #

parameters <- list(mito   = "mitochongenome",
                   chloro = "chlorogenome"  )

# ============================================================================ #
# Prepare data                                                            ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Prepare count matrix                                                   ----
## --------------------------------------------------------------------------- #

### Function
prepare_count_matrix <- function(
    count_matrix_file, count_cells_file, count_genes_file
) {
  # Load matrix
  count_matrix <- count_matrix_file |> Matrix::readMM()
  # Load corresponding cell names
  count_cells <- count_cells_file |> 
    read_csv(col_names = FALSE, show_col_types = FALSE) |> 
    pull()
  # Load corresponding gene names
  count_genes <- count_genes_file |> 
    read_csv(col_names = FALSE, show_col_types = FALSE) |> 
    pull()
  # Rename rows with cells and columns with genes
  rownames(count_matrix) <- count_cells
  colnames(count_matrix) <- count_genes
  # Transpose for seurat compatibility
  count_matrix <- Matrix::t(count_matrix)
  # Output
  count_matrix
}

### Process each section (combined in list)
count_matrix_by_section <- sections$Section_ID |>  
  set_names() |> 
  map(function(section_ID) {
    details <- sections |> filter(Section_ID == section_ID)
    prepare_count_matrix(
      details$Count_matrix, details$Count_cells, details$Count_genes
    )
  })

## --------------------------------------------------------------------------- #
## Load initial cell-types                                                ----
## --------------------------------------------------------------------------- #

### Process for each section (combined in list)
initial_celltypes_by_section <- sections$Section_ID |>  
  set_names() |> 
  map(function(section_ID) {
    # Get file details
    sections |> 
      filter(Section_ID == section_ID) |> 
      pull(Initial_celltypes) |>
      # Load
      read_csv(show_col_types = FALSE)
  })

# ============================================================================ #
# Create seurat                                                           ====
# ============================================================================ #

## Function
prepare_seurat <- function(section_ID, count_matrix, initial_celltypes) {
  
  # Create seurat
  seurat <- CreateSeuratObject(count_matrix             ,
                               project      = section_ID,
                               min.cells    = 1         ,
                               min.features = 1         )
  
  # Calculate percent of mitochondrial/chloroplastic genes
  seurat[["Mito_percent"]]   <- seurat |> 
    PercentageFeatureSet(pattern = parameters$mito)
  seurat[["Chloro_percent"]] <- seurat |> 
    PercentageFeatureSet(pattern = parameters$chloro)
  
  # Add initial cell-types
  seurat <- seurat |>
    AddMetaData(initial_celltypes, col.name = "Initial_celltype")
  
  # Output
  seurat 
}

## Process for each section (combined in list)
seurat_by_section <- count_matrix_by_section |> 
  imap(function(count_matrix, section_ID) {
    # Get initial cell-types
    initial_celltypes <- initial_celltypes_by_section[[section_ID]] |> 
      column_to_rownames("cell_ID")
    # Process
    prepare_seurat(section_ID, count_matrix, initial_celltypes)
  })

# Merge all seurats
seurat <- merge(x            = seurat_by_section[[1]],
                y            = seurat_by_section[-1],
                add.cell.ids = names(seurat_by_section),
                merge.data   = FALSE,
                project      = "wheat-root_STEREO-seq")

# Rename "orig.ident"
colnames(seurat@meta.data)[1] <- "Section_ID"

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

write_rds(seurat, outputs$seurat, compress = "gz")
