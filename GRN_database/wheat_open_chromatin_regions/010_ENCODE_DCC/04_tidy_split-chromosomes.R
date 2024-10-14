# ############################################################################ #
# Tidy split chromosomes
# ############################################################################ #

# As wheat chromosomes were split in order to be compatible with the ENCODE-DCC
# pipeline, peak results are modified here to reflect their original chromosome
# and positions.

# Arguments:
# chromosomes - Correspondence between split and original chromosomes/positions
# narrowPeak  - narrowPeak results from the pipeline
# output      - narrowPeak results with original chromosomes and positions

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
paths$set  <- here("data_processed/010_ENCODE-DCC/wang-2022_AK58_spike_dr")
paths$out  <- here(paths$set, "peak_original-chromosome")

### General
general <- tibble(
  chromosomes = glue("{paths$core}/split-chromosomes_correspondence.txt"),
  narrowPeak  = glue("{paths$set}/peak/overlap_reproducibility/overlap.optimal_peak.narrowPeak.gz"),
  output      = glue("{paths$out}/overlap.optimal_peak.narrowPeak.gz")
)

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Create output directory
if(!dir.exists(paths$out)) dir.create(paths$out, recursive = TRUE)

## --------------------------------------------------------------------------- #
## Prepare data                                                           ----
## --------------------------------------------------------------------------- #

### Load chromosomes
chromosomes <- general$chromosomes |> read_delim(col_types = "ffiiiifi")

### Load peaks
narrowPeak <- general$narrowPeak |>
  read_delim(col_names = c("chrom", "chromStart", "chromEnd", "name", "score",
                           "strand", "signalValue", "pValue", "qValue", "peak"),
             col_types = "fiicnfnnni")

## --------------------------------------------------------------------------- #
## Tidy chromosomes and positions                                         ----
## --------------------------------------------------------------------------- #

narrowPeak <- narrowPeak |>
  # Join with original chromosomes
  left_join(chromosomes |>
              select(Split_name, Split_left, Original_chromosome),
            by = join_by(chrom == Split_name)) |>
  # Adjust positions
  mutate(chromStart = chromStart + Split_left,
         chromEnd   = chromEnd   + Split_left) |>
  # Tidy
  select(-chrom, -Split_left) |>
  rename(chrom = Original_chromosome) |>
  relocate(chrom, .before = 1)

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

narrowPeak |> write_delim(general$output, delim = "\t", col_names = FALSE)
