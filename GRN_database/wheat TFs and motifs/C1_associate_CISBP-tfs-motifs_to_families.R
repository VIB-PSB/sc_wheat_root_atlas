# ############################################################################ #
# Associate CIS-BP TFs with TF families
# ############################################################################ #

# TF IDs from CIS-BP and plantTFDB mostly corresponds to each other.

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
paths$out  <- glue("{paths$root}/data_processed/PlantTFDB")

### Files
files            <- list()
files$tfs_motifs <- glue("{paths$data}/cis-bp_IWGSC-v2.1_TFs-motifs.csv")
files$families   <- glue("{paths$raw}/PlantTFDB/tae_20221108/Tae_TF_list.txt.gz")
files$motifs_avail_uniq <- glue("{paths$data}/available_motifs_unique.csv")

files$tfs_motifs_families        <- glue("{paths$data}/cis-bp_IWGSC-v2.1_TFs-motifs-families.csv")
files$tfs_motifs_families_unique <- glue("{paths$data}/cis-bp_IWGSC-v2.1_TFs-motifs-families_unique.csv")

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Prepare data                                                           ----
## --------------------------------------------------------------------------- #

tfs_motifs <- files$tfs_motifs |> read_csv()
families   <- files$families |>
  read_delim(delim = "\t", col_select = c(Gene_id = Gene_ID, Family))

motifs_avail_uniq <- files$motifs_avail_uniq |> read_csv()

## --------------------------------------------------------------------------- #
## Get TFs with one unique family                                         ----
## --------------------------------------------------------------------------- #

# This could be improved as TFs with multiple families often are often due to
# different naming for the same family.

tfs_with_one_family <- tfs_motifs |>
  # Get unique list of TFs based on original IDs
  select(Gene_id_original, Gene_id_iwgsc_v2.1) |>
  distinct() |>
  # Add families
  left_join(families, by = c("Gene_id_original" = "Gene_id")) |>
  # Remove TFs without families
  filter(!is.na(Family)) |>
  # Get unique list of TFs based on IWGSC v2.1 IDs
  select(-Gene_id_original) |>
  distinct() |>
  # Remove TFs associated with multiple families
  group_by(Gene_id_iwgsc_v2.1) |>
  mutate(families = Family |> n_distinct()) |>
  ungroup() |>
  filter(families == 1) |>
  select(-families)

## --------------------------------------------------------------------------- #
## Associate TFs, motifs and families                                     ----
## --------------------------------------------------------------------------- #

tfs_motifs_families <- tfs_motifs |>
  # Remove redundant rows (should not be any)
  distinct() |>
  # Add unique families
  left_join(tfs_with_one_family, by = "Gene_id_iwgsc_v2.1") |>
  # Remove TFs without motifs nor family
  filter(!(is.na(Motif_id) & is.na(Family))) |>
  # Remove rows without motifs when the corresponding TF has motifs
  group_by(Gene_id_iwgsc_v2.1) |>
  mutate(Motifs    = Motif_id |> n_distinct(na.rm = TRUE),
         Has_motif = Motifs > 0) |>
  ungroup() |>
  filter(!(is.na(Motif_id) & Has_motif)) |>
  select(-c(Motifs, Has_motif))

## --------------------------------------------------------------------------- #
## Filter to unique motifs                                                ----
## --------------------------------------------------------------------------- #

tfs_motifs_families_unique <- tfs_motifs_families |>
  left_join(motifs_avail_uniq |> mutate(Unique = TRUE), by = "Motif_id") |>
  filter(Unique) |>
  select(-Unique)

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

tfs_motifs_families |> write_csv(files$tfs_motifs_families)
tfs_motifs_families_unique |> write_csv(files$tfs_motifs_families_unique)
