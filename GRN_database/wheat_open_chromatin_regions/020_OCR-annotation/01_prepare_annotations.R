# ############################################################################ #
# Prepare genome annotations for OCRs
# ############################################################################ #

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(glue)

# GenomicFeatures and AnnotationDbi (mask too many standard functions)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths      <- list()
paths$core <- here("data_core")
paths$out  <- here("data_processed/020_OCR-annotation")

### General
general <- list(
  # Genome annotations (GFF3)
  annotations = glue("{paths$core}/triticum-aestivum_IWGSC-2.1_simplified.gff3.gz"),
  # Outputs
  out_anno    = glue("{paths$out}/triticum-aestivum_IWGSC-2.1_simplified.sqlite")
)

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## Create output directory
if(!dir.exists(paths$out)) dir.create(paths$out, recursive = TRUE)

## Convert with GenomicFeatures
annotations <- GenomicFeatures::makeTxDbFromGFF(
  file       = general$annotations,
  dataSource = "IWGSC",
  organism   = "Triticum aestivum"
)

## Export
annotations |> AnnotationDbi::saveDb(general$out_anno)
