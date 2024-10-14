# ############################################################################ #
# Select OCRs
# ############################################################################ #

# OCRs are selected in two steps:
# - Some OCRs are redundant as they overlap completely ("overlap.optimal_peak").
#   Only the first one is selected.
# - Some genes are associated with too many OCRs. Those are removed.
# A table of OCRs with their selection status is exported.

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)
library(readxl)

library(furrr)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths            <- list()
paths$core       <- here("data_core")
paths$annotation <- here("data_processed/020_OCR-annotation")

### General
general <- list(
  studies     = glue("{paths$core}/ATAC-seq_studies.xlsx")
)

### Set-specific
sets <- general$studies |>
  read_xlsx(sheet = "Sets") |>
  mutate(
    # Inputs
    in_OCRs = glue("{paths$annotation}/{Set}/OCRs.csv.gz"),
    # Outputs
    out_selection = glue("{paths$annotation}/{Set}/OCRs-selection.csv")
  )

## --------------------------------------------------------------------------- #
## Define parameters                                                      ----
## --------------------------------------------------------------------------- #

parameters <- list(max_length = 10000)

# ============================================================================ #
# Functions                                                               ====
# ============================================================================ #

select_OCR <- function(in_OCRs, out_selection) {

  # Load OCRs with annotation
  OCRs <- in_OCRs |>
    read_csv(col_select = c("OCR", "Chromosome", "Start", "End", "Length", "Gene_id"))

  # Prepare selection table
  selection <- OCRs |>
    select(OCR) |>
    mutate(Reason = NA)

  # Remove redundant OCRs
  OCRs <- OCRs |>
    group_by(Chromosome, Start, End) |>
    slice_head(n = 1) |>
    ungroup()

  # Update selection status
  selection <- selection |>
    mutate(Reason = case_when(!is.na(Reason)    ~ Reason,
                              OCR %in% OCRs$OCR ~ NA,
                              .default          = "redundant"))

  # Get selected genes (based on cumulated OCR length)
  genes_selected <- OCRs |>
    group_by(Gene_id) |>
    summarise(OCRs_length = Length |> sum(), .groups = "drop") |>
    filter(OCRs_length <= parameters$max_length) |>
    pull(Gene_id)

  # Filter to selected genes
  OCRs <- OCRs |>
    filter(Gene_id %in% genes_selected) |>
    select(OCR)

  # Update selection status
  selection <- selection |>
    mutate(Reason = case_when(!is.na(Reason)    ~ Reason,
                              OCR %in% OCRs$OCR ~ NA,
                              .default          = "long-cumulative-length")) |>
    mutate(Selected = is.na(Reason), .before = Reason)

  # Export
  selection |> write_csv(out_selection)

}

# ============================================================================ #
# Process in parallel                                                     ====
# ============================================================================ #

plan(multisession, workers = availableCores() - 2)

# For each set
1:nrow(sets) |>
  future_walk(function(set_row) {
    set_args = sets[set_row, ]

    select_OCR(in_OCRs       = set_args$in_OCRs,
               out_selection = set_args$out_selection)
  })

plan(sequential)
