#!/bin/bash
#$ -N CompareMatrices
#$ -cwd
#$ -l h_vmem=30G

# ##############################################################################
# Compare all motifs with RSAT
# ##############################################################################

# ==============================================================================
# Steps to process manually
# ==============================================================================

# Before starting, import:
# - List of CIS-BP motifs to use ($IN_CISBP_LIST)
# - Individual CIS-BP motifs PWM ($DIR_CISBP/*)

# Once finished, export:
# - CIS-BP motifs converted to transfac format ($OUT_CISBP_TRANSFAC)
# - CIS-BP ID converter due to transfac conversion ($OUT_CISBP_IDS)
# - List of all motifs names ($OUT_MOTIFS)
# - Comparison results ($OUT_RSAT)

# ==============================================================================
# Setup
# ==============================================================================

## Define paths
DIR_MAIN=/group/irc/personal/vincentp
DIR_CISBP=$DIR_MAIN/processing/motifs/CIS-BP/tae_20220908/pwms_all_motifs/
DIR_RSAT=$DIR_MAIN/processing/motifs/RSAT
DIR_OUT=$DIR_RSAT/results

## Define input files
IN_CISBP_LIST=$DIR_RSAT/cis-bp_motifs.csv

## Define output files
OUT_CISBP_MERGED=$DIR_OUT/cis-bp_motifs.cisbp
OUT_CISBP_TRANSFAC=$DIR_OUT/cis-bp_motifs.transfac
OUT_CISBP_IDS=$DIR_OUT/cis-bp_motifs_id-converter.txt

OUT_AVAIL_LIST=$DIR_OUT/cis-bp_available-motifs.csv  # Some motifs are not available (TRANSFAC license)
OUT_RSAT=$DIR_OUT/cis-bp_available-motifs_rsat-comparison  # Creates 3 files

## Load modules
module load rsa-tools

## Create output directory
mkdir -p $DIR_OUT

# ==============================================================================
# Process
# ==============================================================================

## -----------------------------------------------------------------------------
## Prepare CIS-BP motifs
## -----------------------------------------------------------------------------

### Merge all motifs
# IDs are first added using the DE field to preserve them during conversion (no other way found)
echo > $OUT_CISBP_MERGED
awk 'NR>1' $IN_CISBP_LIST | while read MOTIF
do
  # Only if file has a matrix (those from TRANSFAC are empty)
  MOTIF_FILE=$DIR_CISBP/$MOTIF.txt
  if [ `wc -l $MOTIF_FILE | awk '{print $1}'` -ge "2" ]; then
    cat $MOTIF_FILE >> $OUT_CISBP_MERGED
    echo -e "DE\t$MOTIF\n" >> $OUT_CISBP_MERGED
  fi
done

### Convert to transfac format
convert-matrix -from cis-bp -to transfac -return counts,consensus -i $OUT_CISBP_MERGED -o $OUT_CISBP_TRANSFAC

### Get correspondence between original and new IDs (from conversion)
grep -E "ID|DE" $OUT_CISBP_TRANSFAC | awk '$1 == "DE" { print prev, $2} { prev = $2 }' > $OUT_CISBP_IDS

### Get list of available motifs
echo "Motif_id" > $OUT_AVAIL_LIST
awk '$1=="DE" {print $2}' $OUT_CISBP_TRANSFAC >> $OUT_AVAIL_LIST

## -----------------------------------------------------------------------------
## Compare matrices
## -----------------------------------------------------------------------------

compare-matrices -v 1 -format transfac -file $OUT_CISBP_TRANSFAC \
  -strand DR -lth Ncor 1 \
  -return cor,Ncor,match_rank,matrix_id \
  -o $OUT_RSAT


echo "Script finished at $(date)."
