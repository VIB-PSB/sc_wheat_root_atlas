# ############################################################################ #
# Prepare GO files
# ############################################################################ #

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths      <- list()
paths$core <- here("data_core")
paths$data <- here("data_processed")
paths$out  <- here(paths$data)

### General
general <- list(
  GOs     = glue("{paths$core}/go_tae_bp.csv.gz"),
  out_GOs = glue("{paths$out}/tae_GOs.tsv"))

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Load GOs
GOs <- general$GOs |>
  read_csv()

## Adjust for MINI-EX
GOs <- GOs |>
  select(go_id, gene_id, evidence, description)

## Export
GOs |> write_tsv(general$out_GOs, col_names = FALSE)
