#!/bin/bash
set -e

# ##############################################################################
# Map motif with FIMO
# ##############################################################################

# Each motif are mapped on regulatory regions using FIMO.
# Motifs are first prepared for mapping, then split into a list of n motifs and 
# submitted for mapping as n SGE jobs.
# A background model is also created.

# ==============================================================================
# Steps to process manually
# ==============================================================================

# Before starting:
# - Import the run script - $SCRIPT_RUN
# - Import the motifs PWMs - $DIR_PWMs
# - Import the list of unique motifs (with header) - $IN_MOTIFS_LIST

# Once finished:
# - Export 

# Set already processed:
# - pei-2022_AK58_root_2wo
# - wang-2022_AK58_spike_dr

# Note: motifs are prepared for fimo each time to include the set background.

# ==============================================================================
# Setup
# ==============================================================================

IN_SET=pei-2022_AK58_root_2wo

## Define paths
DIR_MAIN=/group/irc/personal/vincentp/processing

DIR_MOTIFS=$DIR_MAIN/motifs
DIR_PWMs=$DIR_MOTIFS/CIS-BP/tae_20220908/pwms_all_motifs

DIR_MAPPING=$DIR_MAIN/motif-mapping
DIR_OCRs=$DIR_MAPPING/OCRs

DIR_OUT=$DIR_MAPPING/mapping/$IN_SET/fimo
DIR_OUT_MOTIFS=$DIR_OUT/motifs
DIR_OUT_MAPPING=$DIR_OUT/mapping
DIR_OUT_LOGS=$DIR_OUT/logs

## Define input files
IN_OCRs=$DIR_OCRs/$IN_SET\_overlap.optimal_peak.fasta
IN_MOTIFS_LIST=$DIR_MOTIFS/cis-bp_available-motifs_unique.csv

SCRIPT_RUN=$DIR_MAPPING/scripts/02b_run_fimo_map-motifs.bash

## Define output files
OUT_BACKGROUND=$DIR_OUT/background.txt

## Define parameters
N_JOBS=30  # Number of jobs to run

## Create output directories
mkdir -p $DIR_OUT
mkdir -p $DIR_OUT_MOTIFS
mkdir -p $DIR_OUT_MAPPING
mkdir -p $DIR_OUT_LOGS
cd $DIR_OUT_LOGS

## Load modules (meme 5.4.1 for background modeling only)
module load meme/x86_64/5.4.1

# ==============================================================================
# Submit
# ==============================================================================

echo "Starting processing set $IN_SET."

## -----------------------------------------------------------------------------
## Get background model (using meme utility)
## -----------------------------------------------------------------------------

# Note: sequences should be biologically similar to the scanned sequences.
fasta-get-markov $IN_OCRs $OUT_BACKGROUND

## -----------------------------------------------------------------------------
## Prepare motifs (using meme utility)
## -----------------------------------------------------------------------------

awk 'NR>1' $IN_MOTIFS_LIST | while read MOTIF; do
	# Prepare matrix
	tail -n +2 $DIR_PWMs/$MOTIF.txt | cut -f 2- | \
	# Convert to MEME format
	matrix2meme -bg $OUT_BACKGROUND | \
	# Add motif name
	sed "s/MOTIF 1 .*/MOTIF $MOTIF/" > $DIR_OUT_MOTIFS/$MOTIF.txt
done

## -----------------------------------------------------------------------------
## Map each motif on OCRs
## -----------------------------------------------------------------------------

### List motifs to be used for each job
N_MOTIFS=$(ls $DIR_OUT_MOTIFS/ | wc -l)
N_MOTIFS_PER_JOB=$(echo $((N_MOTIFS / N_JOBS)))
ls $DIR_OUT_MOTIFS | xargs -n $N_MOTIFS_PER_JOB > job_list

### Launch run script (n jobs with n motifs)
cat job_list | while read MOTIFS; do
  qsub $SCRIPT_RUN  \
    --motifs "$MOTIFS" \
	--dir_motifs $DIR_OUT_MOTIFS \
	--sequences $IN_OCRs \
	--dir_out $DIR_OUT_MAPPING
done

rm job_list

echo "Script finished ($(date))."
