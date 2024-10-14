# ############################################################################ #
# Annotate OCRs
# ############################################################################ #

# Take a narrowPeak file and annotate it with ChIPseeker.

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(here)
here::i_am(".here")

library(tidyverse)
library(readxl)
library(glue)

library(AnnotationDbi)
library(ChIPseeker)

library(furrr)

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths             <- list()
paths$core        <- here("data_core")
paths$narrowPeak  <- here("data_processed/010_ENCODE-DCC")
paths$annotations <- here("data_processed/020_OCR-annotation")
paths$out         <- paths$annotations

### General
general <- list(
  studies     = glue("{paths$core}/ATAC-seq_studies.xlsx"),
  annotations = glue("{paths$annotations}/triticum-aestivum_IWGSC-2.1_simplified.sqlite")
)

### Set-specific
sets <- general$studies |>
  read_xlsx(sheet = "Sets") |>
  mutate(
    # narrowPeak file
    narrowPeak = glue("{paths$narrowPeak}/{Set}/",
                      "peak_original-chromosome/overlap.optimal_peak.narrowPeak.gz"),
    # Outputs
    out_dir  = glue("{paths$out}/{Set}"),
    out_OCRs = glue("{out_dir}/OCRs.csv.gz"),
    out_anno = glue("{out_dir}/OCRs-annotations.rds")
  )

# ============================================================================ #
# Functions                                                               ====
# ============================================================================ #

annotate_OCRs <- function(set, narrowPeak, out_dir, out_OCRs, out_anno) {

  # -------------------------------------------------------------------------- #
  # Prepare data                                                          ----
  # -------------------------------------------------------------------------- #

  ## Load annotations
  annotations <- AnnotationDbi::loadDb(general$annotations)

  ## Load OCRs for custom use
  OCRs_custom <- narrowPeak |>
    # Load
    read_delim(col_names = c("Chromosome", "Start", "End", "OCR", "Score",
                             "Strand", "Signal", "p_value", "q_value",
                             "Peak_position"),
               col_types = "fiicnfnnni") |>
    relocate(OCR, .before = 1) |>
    # Add length
    mutate(Length = End - Start, .after = End)  # End base is not included

  ## Load OCRs for ChIPseeker
  OCRs_ChIPseeker <- narrowPeak |> readPeakFile()

  # -------------------------------------------------------------------------- #
  # Process                                                               ----
  # -------------------------------------------------------------------------- #

  ## Annotate
  OCRs_anno <- annotatePeak(peak             = OCRs_ChIPseeker,
                            TxDb             = annotations,
                            tssRegion        = c(-5000, 5000),
                            addFlankGeneInfo = TRUE,
                            flankDistance    = 50000)

  ## Tidy annotations for custom use
  anno_tidy <- OCRs_anno@anno@elementMetadata@listData |>
    as_tibble() |>
    dplyr::select(OCR          = V4,
                  Annotation   = annotation,
                  Gene_id      = geneId,
                  TSS_distance = distanceToTSS) |>
    # Tidy annotations
    mutate(Annotation = Annotation |>
             str_replace("Exon .*", "Exon") |>
             str_replace("Intron .*", "Intron") |>
             str_replace("Promoter \\(<=1kb\\)", "Promoter (0-1kb)"))

  OCRs_custom <- OCRs_custom |>
    left_join(anno_tidy, by = "OCR")

  # -------------------------------------------------------------------------- #
  # Export                                                                ----
  # -------------------------------------------------------------------------- #

  if(!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  OCRs_custom |> write_csv(out_OCRs)
  OCRs_anno   |> write_rds(out_anno, compress = "gz")
}

# ============================================================================ #
# Process in parallel                                                     ====
# ============================================================================ #

plan(multisession, workers = availableCores() - 2)

# For each set
1:nrow(sets) |>
  future_walk(function(set_row) {
    set_args = sets[set_row, ]
    annotate_OCRs(set        = set_args$Set,
                  narrowPeak = set_args$narrowPeak,
                  out_dir    = set_args$out_dir,
                  out_OCRs   = set_args$out_OCRs,
                  out_anno   = set_args$out_anno)
  })

plan(sequential)
