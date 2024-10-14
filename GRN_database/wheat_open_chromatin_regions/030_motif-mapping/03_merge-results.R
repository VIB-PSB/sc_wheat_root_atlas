# ############################################################################ #
# Merge all mapping results
# ############################################################################ #

# All mapping results are merged into a single table.
# Some modification are being made.

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
paths         <- list()
paths$core    <- here("data_core")
paths$mapping <- here("data_processed/030_motif-mapping")

### General
general <- list(
  studies = glue("{paths$core}/ATAC-seq_studies.xlsx")
)

### Set-specific
sets <- general$studies |>
  read_xlsx(sheet = "Sets") |>
  mutate(
    # Inputs
    in_mapping = glue("{paths$mapping}/{Set}/fimo/mapping"),
    # Outputs
    out_mapping = glue("{paths$mapping}/{Set}/merged-results.rds")
  )

# ============================================================================ #
# Functions                                                               ====
# ============================================================================ #

merge_results <- function(in_mapping, out_mapping) {

  # Prepare parallel
  plan(multisession, workers = availableCores() - 2)

  # Merge results in a list
  mapping_by_motif <- in_mapping |>

    # For each motif-mapping file
    list.files(recursive = TRUE, full.names = TRUE) |>
    future_map(function(file){

      # Return NULL for un-mapped motifs
      if(file.info(file)["size"] < 1000) {
        return(NULL)
      }

      # Count number of data rows for skipping last rows
      # (using comment = "#" directly in read_delim() fails sometimes)
      nrows = file |>
        count.fields(sep = "\t", skip = 1, comment.char = "#") |>
        length()

      # Process
      file |>
        # Load
        read_delim(delim = "\t",
                   col_types  = "fcfiifnnnc",
                   col_select = -c(motif_alt_id, matched_sequence),
                   n_max = nrows) |>
        # Tidy
        rename(Motif_id = motif_id,
               OCR      = sequence_name,
               Start    = start,
               Stop     = stop,
               Strand   = strand,
               Score    = score,
               p_value  = `p-value`,
               q_value  = `q-value`)
    }) |>
    # Remove motifs without mapping results
    discard(is.null)

  plan(sequential)

  # Convert list to tibble
  mapping <- mapping_by_motif |> list_rbind()

  # Export
  mapping |> write_rds(out_mapping, compress = "gz")
}

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

# For each set
1:nrow(sets) |>
  walk(function(set_row) {
    set_args = sets[set_row, ]

    merge_results(in_mapping  = set_args$in_mapping,
                  out_mapping = set_args$out_mapping)
  })
