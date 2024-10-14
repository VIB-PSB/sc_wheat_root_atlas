# ############################################################################ #
# Select best IWGSC TFs
# ############################################################################ #

# The best hit for each TF is detected, and the corresponding motifs are
# associated with it.

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
paths$core <- glue("{paths$root}/data_core")
paths$data <- glue("{paths$root}/data_processed/CIS-BP")
paths$out  <- paths$data

paths$cisbp_db <- glue("{paths$raw}/CIS-BP/tae_20220908")

### Files
files                <- list()
files$cisbp_blast    <- glue("{paths$data}/cis-bp_IWGSC-v2.1_TFs-blast.txt.gz")
files$cisbp_tfs      <- glue("{paths$cisbp_db}/prot_seq.txt")
files$cisbp_motifs   <- glue("{paths$cisbp_db}/TF_Information_all_motifs.txt")

files$out_besthit    <- glue("{paths$out}/cis-bp_IWGSC-v2.1_TFs-blast_best-hit.csv")
files$out_tfs_motifs <- glue("{paths$out}/cis-bp_IWGSC-v2.1_TFs-motifs.csv")

## --------------------------------------------------------------------------- #
## Define parameters                                                      ----
## --------------------------------------------------------------------------- #

parameters <- list(evalue = 1e-50, pident = 90)

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Prepare data                                                           ----
## --------------------------------------------------------------------------- #

### Load blast results
cisbp_blast <- files$cisbp_blast |>
  read_delim() |>
  rename(TF_id = qaccver, Peptide_id_iwgsc_v2.1 = saccver)

### Load TF sequences
cisbp_tfs <- files$cisbp_tfs |>
  read_delim() |>
  select(TF_id = TF_ID, Gene_id_original = Gene_ID) |>
  # Add chromosome based on gene id
  mutate(Chromosome_original = Gene_id_original |>
           str_remove(".*_WHEAT$") |>
           str_remove("^Traes_|^TRAES|^TraesCS") |>
           str_extract("^[:digit:]{0,1}[ABDU]"))

### Load motifs
cisbp_motifs <- files$cisbp_motifs |>
  read_delim(na = ".") |>
  select(TF_id            = TF_ID,
         Motif_id         = Motif_ID,
         Gene_id_original = DBID...6,
         TF_status        = TF_Status)

# ============================================================================ #
# Process BLAST results                                                   ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Filter blast results                                                   ----
## --------------------------------------------------------------------------- #

# Filter blast results with evalue and pident thresholds.

cisbp_blast_filter <- cisbp_blast |>
  filter(evalue <= parameters$evalue,
         pident >= parameters$pident) |>
  # Add original gene ID
  left_join(cisbp_tfs, by = "TF_id") |>
  # Convert IWGSC v2.1 peptide ID to gene ID and add chromosome
  mutate(Gene_id_iwgsc_v2.1    = Peptide_id_iwgsc_v2.1 |> str_remove("\\..*"),
         Chromosome_iwgsc_v2.1 = Gene_id_iwgsc_v2.1 |>
           str_remove("TraesCS") |>
           str_extract("^[:digit:]{0,1}[ABDU]")) |>
  # Tidy
  select(-Peptide_id_iwgsc_v2.1) |>
  relocate(Gene_id_original, Chromosome_original,
           Gene_id_iwgsc_v2.1, Chromosome_iwgsc_v2.1,
           .after = TF_id)

## --------------------------------------------------------------------------- #
## Select best blast hit                                                  ----
## --------------------------------------------------------------------------- #

# Sequential selection on:
#   - minimum evalue
#   - maximum bitscore
#   - maximum pident
#   - same chromosome as original
#   - random selection

cisbp_blast_best <- cisbp_blast_filter |>
  # Prepare
  mutate(Selected_on = NA_character_) |>
  group_by(Gene_id_original) |>
  # Selection on evalue
  slice_min(evalue) |>
  mutate(Count       = n(),
         Selected_on = case_when(Count == 1 ~ replace_na(Selected_on, "evalue"))) |>
  # Selection on bitscore
  slice_max(bitscore) |>
  mutate(Count       = n(),
         Selected_on = case_when(Count == 1 ~ replace_na(Selected_on, "bitscore"))) |>
  # Selection on pident
  slice_max(pident) |>
  mutate(Count       = n(),
         Selected_on = case_when(Count == 1 ~ replace_na(Selected_on, "pident"))) |>
  # Selection on same chromosome as original
  mutate(Same = (Chromosome_original == Chromosome_iwgsc_v2.1) |> replace_na(FALSE)) |>
  slice_max(Same) |>
  mutate(Count       = n(),
         Selected_on = case_when(Count == 1 ~ replace_na(Selected_on, "chromosome"))) |>
  # Selection on random
  slice_sample() |>
  mutate(Selected_on = replace_na(Selected_on, "random")) |>
  # Tidy
  ungroup() |>
  select(-Count, -Same)

## --------------------------------------------------------------------------- #
## Associate TF with motifs                                               ----
## --------------------------------------------------------------------------- #

cisbp_motifs_fixed <- cisbp_blast_best |>
  select(TF_id, Gene_id_original, Gene_id_iwgsc_v2.1) |>
  left_join(cisbp_motifs, by = c("TF_id", "Gene_id_original"))

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

write_csv(cisbp_blast_best  , files$out_besthit   )
write_csv(cisbp_motifs_fixed, files$out_tfs_motifs)
