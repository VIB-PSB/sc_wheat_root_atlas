#!/bin/bash

# ##############################################################################
# Build genome
# ##############################################################################

# This script submits a job to the cluster.
# Important: genome path was hard coded in the "run" script.

# ==============================================================================
# Prepare environment
# ==============================================================================

## Define paths
DIR_MAIN=/group/irc/personal/vincentp
DIR_REF=$DIR_MAIN/references/triticum-aestivum_IWGSC-2.1_MtCp_split
DIR_DESTINATION=$DIR_REF/ENCODE-DCC

## Define URLs (tab delimited: URL new-name)
SCRIPT_RUN=$DIR_REF/01b_run_build_genome_data.sh

## acticate environment
source /software/shared/apps/x86_64/anaconda/3/etc/profile.d/conda.sh
conda activate encd-atac

# ==============================================================================
# Execute
# ==============================================================================

qsub -V -cwd $SCRIPT_RUN "triticum-aestivum_IWGSC-2.1_MtCp_split" $DIR_DESTINATION
