# ############################################################################ #
# Get list of motifs
# ############################################################################ #

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(tidyverse)
library(glue)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths      <- list()
paths$root <- "D:/Master/Workspace/Resources_Omics/triticum-aestivum_TFs-and-motifs"
setwd(paths$root)

paths$raw  <- glue("{paths$root}/data_raw")
paths$data <- glue("{paths$root}/data_processed/CIS-BP")
paths$out  <- paths$data

paths$cisbp_db <- glue("{paths$raw}/CIS-BP/tae_20220908/pwms_all_motifs")

### Files
out_motifs <- glue("{paths$out}/cis-bp_motifs.csv")

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Get all motifs from their PWMs
motifs <- paths$cisbp_db |>
  list.files() |>
  str_remove(".txt") |>
  tibble(Motif_id = _)

## Get all motifs with their PWMs file size (could be used one day)
# Some motifs are not available as they are under TRANSFAC license.
# motifs <- paths$cisbp_db |>
#   list.files(full.names = TRUE) |>
#   set_names(nm = ~ . |> str_remove_all(".*/|.txt")) |>
#   map(function(file) {
#     file |> file.size() |> tibble(size = _)
#   }) |>
#   list_rbind(names_to = "Motif_id")

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

motifs |> write_csv(out_motifs)
