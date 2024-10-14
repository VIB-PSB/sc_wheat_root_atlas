# ############################################################################ #
# Prepare mapping files
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
paths$out  <- here(paths$data, "motif-mapping")

### General
general <- list(
  # Inputs
  mappings   = glue("{paths$core}/mappings.csv"),
  TF2fam2mot = glue("{paths$data}/tae_TF2fam2mot.tsv"))

### Mapping-specific
mappings <- general$mappings |>
  read_csv() |>
  mutate(out_mapping = glue("{paths$out}/tae_{Set}_motifMapping.out.gz"))

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Create output directory
if(!dir.exists(paths$out)) dir.create(paths$out, recursive = TRUE)

## --------------------------------------------------------------------------- #
## Prepare general data                                                   ----
## --------------------------------------------------------------------------- #

### Prepare TF2fam2mot
TF2fam2mot <- general$TF2fam2mot |>
  read_tsv(col_names = c("Gene_id", "TF_family", "TF_status", "Motif_ids")) |>
  filter(!is.na(Motif_ids))

### Get list of motifs
motifs <- TF2fam2mot |>
  pull(Motif_ids) |>
  str_split(",") |>
  unlist() |>
  unique() |>
  sort()

## --------------------------------------------------------------------------- #
## Process mapping                                                        ----
## --------------------------------------------------------------------------- #

# For each mapping set
for(row in 1:nrow(mappings)) {
  args_mapping     <- mappings[[row, "Location"]]
  args_out_mapping <- mappings[[row, "out_mapping"]]

  mapping <- args_mapping |>
    # Load
    read_csv() |>
    select(Motif_id, Gene_id) |>
    distinct() |>
    # Filter to selected motifs
    filter(Motif_id %in% motifs)

  ## Export
  mapping |>
    write_tsv(args_out_mapping, col_names = FALSE)
}
