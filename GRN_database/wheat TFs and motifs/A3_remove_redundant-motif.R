# ############################################################################ #
# Remove redundant motifs
# ############################################################################ #

# Using the results from RSAT "compare-matrices", redundant motifs are removed.
# Motifs are considered redundant based on normalized correlation (Ncor).

# ============================================================================ #
# Setup                                                                   ====
# ============================================================================ #

## Load libraries
library(tidyverse)
library(glue)

# igraph required but not loaded

## --------------------------------------------------------------------------- #
## Define inputs/outputs                                                  ----
## --------------------------------------------------------------------------- #

### Paths
paths      <- list()
paths$root <- "D:/Master/Workspace/Resources_Omics/triticum-aestivum_TFs-and-motifs"
setwd(paths$root)

paths$raw  <- glue("{paths$root}/data_raw/CIS-BP/tae_20220908/")
paths$data <- glue("{paths$root}/data_processed/CIS-BP")
paths$out  <- paths$data

### Files
files <- list()

files$cisbp_ids <- glue("{paths$data}/cis-bp_motifs_id-converter.txt")
files$transfac  <- glue("{paths$data}/cis-bp_motifs.transfac")
files$rsat      <- glue("{paths$data}/cis-bp_available-motifs_rsat-comparison.tab")

files$TFs <- glue("{paths$raw}/TF_Information_all_motifs.txt")

files$out_redundant <- glue("{paths$out}/cis-bp_available-motifs_rsat-redundant.csv")
files$out_unique    <- glue("{paths$out}/cis-bp_available-motifs_unique.csv")

## --------------------------------------------------------------------------- #
## Define parameters                                                      ----
## --------------------------------------------------------------------------- #

threshold_Ncor <- 1  # Note: already filtered during motifs comparison.

# ============================================================================ #
# Process                                                                 ====
# ============================================================================ #

## --------------------------------------------------------------------------- #
## Prepare data                                                           ----
## --------------------------------------------------------------------------- #

## Load ID converter
cisbp_ids <- files$cisbp_ids |>
  read_delim(col_names = c("ID", "Motif_id"), delim = " ")

## Load motifs with TRANSFAC format to get consensus sequence
motifs_consensus <- files$transfac |>
  read_tsv(col_names = "Value") |>
  filter(str_starts(Value, "DE") | str_detect(Value, "consensus.strict:")) |>
  mutate(Value = Value |>
           str_replace("DE[^M]*", "Motif_id ") |>
           str_replace(".*consensus.strict:", "Consensus")) |>
  separate(col = Value, into = c("Type", "Value"), sep = " ") |>
  mutate(Set = rep(1:(n()/2), each = 2)) |>
  pivot_wider(id_cols = Set, names_from = Type, values_from = Value) |>
  select(-Set)

## Load RSAT results
rsat_results <- files$rsat |>
  read_tsv(comment=";") |>
  rename(id1 = "#id1") |>
  # Add motif ID
  left_join(cisbp_ids |> rename(Motif_id1 = Motif_id), by = c("id1" = "ID")) |>
  left_join(cisbp_ids |> rename(Motif_id2 = Motif_id), by = c("id2" = "ID")) |>
  relocate(Motif_id1, Motif_id2, .after = id2) |>
  select(-c(id1, id2, name1, name2)) |>
  # Add consensus sequence
  left_join(motifs_consensus |> rename(Consensus1 = Consensus),
            by = c("Motif_id1" = "Motif_id")) |>
  left_join(motifs_consensus |> rename(Consensus2 = Consensus),
            by = c("Motif_id2" = "Motif_id")) |>
  relocate(Consensus1, Consensus2, .after = Motif_id2)

## Load TFs
TFs <- files$TFs |>
  read_delim(na = ".") |>
  select(TF_id            = TF_ID,
         Motif_id         = Motif_ID,
         Gene_id_original = DBID...6,
         TF_status        = TF_Status) |>
  select(TF_id, Motif_id) |>
  drop_na() |>
  distinct()

## --------------------------------------------------------------------------- #
## Remove redundant motifs
## --------------------------------------------------------------------------- #

# Notes:
# - The weights are slightly different.
# - Motifs are associated to several TFs.
# - The only available with direct evidence is not redundant (manual check).

## Find redundant motifs
rsat_redundant <- rsat_results |>
  filter(Motif_id1 != Motif_id2,
         Ncor >= threshold_Ncor)

## Create an undirected graph and extract sets of redundant motifs
rsat_redundant_graph <- rsat_redundant |>
  select(Motif_id1, Motif_id2) |>
  igraph::graph_from_data_frame(directed = FALSE)
rsat_redundant_sets <- rsat_redundant_graph |>
  igraph::clusters() |>
  _$membership |>
  enframe(name = "Motif_id", value = "Motif_set")

## Add number of TFs for each motif
rsat_redundant_sets <- rsat_redundant_sets |>
  left_join(TFs |>
              group_by(Motif_id) |>
              summarise(TFs = TF_id |> n_distinct()),
            by = "Motif_id")

## Select one motif in redundant set
rsat_redundant_keep <- rsat_redundant_sets |>
  group_by(Motif_set) |>
  # Select motif with most TFs
  slice_max(TFs, with_ties = TRUE) |>
  # If several, select the first one, alphabetically
  arrange(Motif_id, .by_group = TRUE) |>
  slice_head(n = 1) |>
  # ...
  pull(Motif_id)

## Get motifs to remove
rsat_redundant_remove <- rsat_redundant_sets |>
  filter(!Motif_id %in% rsat_redundant_keep) |>
  pull(Motif_id)

## Remove redundant motifs
motifs_unique <- motifs_consensus |>
  select(Motif_id) |>
  filter(!(Motif_id %in% rsat_redundant_remove)) |>
  arrange(Motif_id)
# Note: if using multiple databases, the origin of the motif could be added.

# ============================================================================ #
# Export                                                                  ====
# ============================================================================ #

rsat_redundant |> write_csv(files$out_redundant)
motifs_unique  |> write_csv(files$out_unique)
