#!/bin/bash
#$ -N extractSequences
#$ -cwd
#$ -l h_vmem=10G
set -e

# ==============================================================================
# Steps to process manually
# ==============================================================================

# Before starting:
# - Index reference genome for samtools - samtools faidx $IN_REF
# - Import the narrowPeak file, named with ATAC-seq set in prefix

# Set already processed:
# - pei-2022_AK58_root_2wo
# - wang-2022_AK58_spike_dr

# ==============================================================================
# Setup
# ==============================================================================

IN_SET=pei-2022_AK58_root_2wo

## Define paths
DIR_MAIN=/group/irc/personal/vincentp
DIR_REF=$DIR_MAIN/references/triticum-aestivum_IWGSC-2.1_MtCp
DIR_DATA=$DIR_MAIN/processing/motif-mapping/OCRs

## Define input files
IN_REF=$DIR_REF/triticum-aestivum_IWGSC-2.1_MtCp.fa
IN_PEAK=$DIR_DATA/$IN_SET\_overlap.optimal_peak.narrowPeak.gz

## Define output files
OUT_SEQ=$DIR_DATA/$IN_SET\_overlap.optimal_peak.fasta

## Load modules
module load samtools

# ==============================================================================
# Process
# ==============================================================================

echo "Starting processing set $IN_SET."

cd $DIR_DATA

## Prepare peak names and coordinates as chr:start-end (corrected for samtools)
zcat $IN_PEAK | awk '{print ">" $4}' > names
zcat $IN_PEAK | awk '{print $1 ":" $2+1 "-" $3}' > coordinates

## Extract sequence
samtools faidx $IN_REF -r coordinates > $OUT_SEQ

## Add gene name
awk '/^>/{getline < "names"} {print $0}' $OUT_SEQ > tmp && mv tmp $OUT_SEQ

rm coordinates names

echo "Script finished ($(date))."

