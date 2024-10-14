# ############################################################################ #
# Prepare TF2fam2mot
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
paths$out  <- here("data_processed")

### General
general <- list(
  # Inputs
  motifs_selected = glue("{paths$core}/cis-bp_available-motifs_unique.csv"),
  TFs_motifs      = glue("{paths$core}/cis-bp_IWGSC-v2.1_TFs-motifs.csv"),
  TFs_families    = glue("{paths$core}/evans-2022_TFs.csv"),
  # Outputs
  out_TF2fam2mot  = glue("{paths$out}/tae_TF2fam2mot.tsv"),
  out_TFs         = glue("{paths$out}/tae_TF_list.tsv")
)

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Prepare data                                                           ----
## --------------------------------------------------------------------------- #

### Prepare selected motifs
motifs_selected <- general$motifs_selected |>
  # Load
  read_csv() |>
  distinct() |>
  pull(Motif_id)

### Prepare transcription factors associated with motifs
TFs_motifs <- general$TFs_motifs |>
  # Load
  read_csv() |>
  select(Gene_id = Gene_id_iwgsc_v2.1, Motif_id, TF_status) |>
  # Remove useless rows
  drop_na() |>
  distinct() |>
  # Filter to selected motifs |>
  filter(Motif_id %in% motifs_selected) |>
  # Adjust status
  mutate(TF_status = case_when(TF_status %in% c("D", "I") ~ "Y",
                               .default                   = "Incorrect")) |>
  distinct() |>
  # Concatenate motif IDs
  arrange(Gene_id, Motif_id) |>
  group_by(Gene_id, TF_status) |>
  summarise(Motif_ids = Motif_id |> str_flatten(collapse = ",")) |>
  ungroup()

### Prepare full list of transcription factors with families
TFs_families <- general$TFs_families |>
  # Load
  read_csv() |>
  select(Gene_id = Gene_ID_v2.1, TF_family) |>
  # Remove useless rows
  filter(!is.na(Gene_id)) |>
  distinct() |>
  # Adjust families
  mutate(TF_family = TF_family |> replace_na("Unknown"))

## --------------------------------------------------------------------------- #
## Create TF2fam2mot                                                      ----
## --------------------------------------------------------------------------- #

TF2fam2mot <- TFs_families |>
  # Add motifs to full list of TFs with families
  left_join(TFs_motifs, by = "Gene_id") |>
  # Adjust status
  mutate(TF_status = TF_status |> replace_na("N")) |>
  # Tidy
  distinct() |>
  arrange(Gene_id)

## --------------------------------------------------------------------------- #
## Export                                                                 ----
## --------------------------------------------------------------------------- #

### TF2fam2mot
TF2fam2mot |>
  write.table(general$out_TF2fam2mot,
              sep       = "\t"  ,
              quote     = FALSE ,
              row.names = FALSE ,
              col.names = FALSE ,
              na        = ""    )

# List of TFs
TF2fam2mot |>
  select(Gene_id) |>
  write.table(general$out_TFs,
              sep       = "\t"  ,
              quote     = FALSE ,
              row.names = FALSE ,
              col.names = FALSE ,
              na        = ""    )

