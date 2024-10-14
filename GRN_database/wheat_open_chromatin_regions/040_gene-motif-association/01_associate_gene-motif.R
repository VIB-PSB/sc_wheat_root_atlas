# ############################################################################ #
# Associate motifs to genes
# ############################################################################ #

# Motifs mapped to selected OCRs are associated with the corresponding genes.

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)
library(readxl)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths         <- list()
paths$core    <- here("data_core")
paths$OCRs    <- here("data_processed/020_OCR-annotation")
paths$mapping <- here("data_processed/030_motif-mapping")
paths$out     <- here("data_processed/040_gene-motif-association")

### General
general <- list(studies = glue("{paths$core}/ATAC-seq_studies.xlsx"))

### Set-specific
sets <- general$studies |>
  read_xlsx(sheet = "Sets") |>
  mutate(
    # Inputs
    in_OCRs          = glue("{paths$OCRs}/{Set}/OCRs.csv.gz"),
    in_OCR_selection = glue("{paths$OCRs}/{Set}/OCRs-selection.csv"),
    in_mapping       = glue("{paths$mapping}/{Set}/merged-results.rds"),
    # Outputs
    out_dir          = glue("{paths$out}/{Set}"),
    out_genes_motifs = glue("{out_dir}/genes-motifs.csv.gz")
  )

# ============================================================================ #
# Functions                                                               ====
# ============================================================================ #

associate_gene_motif <- function(in_OCR_selection, in_OCRs, in_mapping,
                                 out_dir, out_genes_motifs) {

  ## ------------------------------------------------------------------------- #
  ## Prepare data                                                         ----
  ## ------------------------------------------------------------------------- #

  ### Load and prepare selected OCRs
  OCRs_selected <- in_OCR_selection |>
    read_csv() |>
    filter(Selected) |>
    pull(OCR)

  ### Load and prepare OCRs
  OCRs <- in_OCRs |>
    read_csv(col_select = c(OCR, Gene_id)) |>
    # Filter to selected OCRs
    filter(OCR %in% OCRs_selected)

  ### Load and prepare mapping results
  mapping <- in_mapping |>
    read_rds() |>
    select(Motif_id, OCR) |>
    # Filter to selected OCRs
    filter(OCR %in% OCRs_selected) |>
    # Remove duplicates
    distinct()

  ## ------------------------------------------------------------------------- #
  ## Associate motifs to genes                                            ----
  ## ------------------------------------------------------------------------- #

  genes_motifs <- OCRs |>
    left_join(mapping, by = "OCR") |>
    select(Gene_id, Motif_id) |>
    distinct()

  ## ------------------------------------------------------------------------- #
  ## Export                                                               ----
  ## ------------------------------------------------------------------------- #

  if(!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  genes_motifs |> write_csv(out_genes_motifs)
}

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

# For each set
1:nrow(sets) |>
  walk(function(set_row) {
    set_args = sets[set_row, ]
    associate_gene_motif(
      in_OCR_selection = set_args$in_OCR_selection,
      in_OCRs          = set_args$in_OCRs,
      in_mapping       = set_args$in_mapping,
      out_dir          = set_args$out_dir,
      out_genes_motifs = set_args$out_genes_motifs
    )
  })

