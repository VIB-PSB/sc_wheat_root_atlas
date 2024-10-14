#!/bin/bash

# ##############################################################################
# BLAST CIS-BP TFs to IWGSC v2.1
# ##############################################################################

# Transcription factors from the CIS-BP have gene IDs that are not compatible
# with the IWGSC v2.1 reference genome. Here, corresponding IDs are retrieved by
# BLASTing CIS-BP TFs against the IWGSC v2.1 peptides.

# The following commands must be run manually.

# Before starting, the following files must be made available:
# - Reference peptide fasta ($REF)
# - CIS-BP TFs sequence ($TF_TXT)

# Once finished import the final file for further processing ($TF_BLAST)

# ==============================================================================
# Variables
# ==============================================================================

## Input
DIR_MAIN=/group/irc/personal/vincentp
DIR_REF=$DIR_MAIN/references/triticum-aestivum_IWGSC-2.1
DIR_TFs=$DIR_MAIN/processing/blast-TFs
DIR_OUT=$DIR_TFs

IN_REF_HC=$DIR_REF/iwgsc_refseqv2.1_annotation_200916_HC_pep.valid.fasta.gz
IN_REF_LC=$DIR_REF/iwgsc_refseqv2.1_annotation_200916_LC_pep.valid.fasta.gz

IN_TFs=$DIR_TFs/prot_seq.txt

## Output
REF_ALL=iwgsc_refseqv2.1_annotation_200916_all_pep.valid.fasta

TF_FASTA=cis-bp_tf-seq.fasta
TF_SPLIT=$TF_FASTA.split
TF_BLAST=cis-bp_IWGSC-v2.1_TFs-blast.txt

JOBS=15

# ==============================================================================
# Prepare files
# ==============================================================================

cd $DIR_OUT

## Convert sequences to fasta
awk 'NR>1, OFS="\n" {print ">"$1, $10}' $IN_TFs > $TF_FASTA

## Split fasta into n files (using ceiling and with even number of lines).
split -l $(((`wc -l < $TF_FASTA` / 2 + $JOBS - 1) / $JOBS * 2)) $TF_FASTA $TF_SPLIT -d

# ==============================================================================
# Blast
# ==============================================================================

cd $DIR_OUT

module load blast+

## Create BLAST database (on SGE)
cat $IN_REF_HC $IN_REF_LC | gunzip > $REF_ALL
qsub -b y -V -cwd -l h_vmem=10G makeblastdb -in $REF_ALL -dbtype "prot"

## BLAST each split (on SGE)
BLAST_DIR=blast
mkdir $BLAST_DIR
for SPLIT in *split*; do
  BLAST_NAME=$(basename ${SPLIT/.fasta/})_blast.txt
  qsub -b y -V -cwd -l h_vmem=10G -N BLAST \
  blastp -db $REF_ALL -query $SPLIT -out $BLAST_DIR/$BLAST_NAME -outfmt 6
done

## Merge results with header
echo -e "qaccver\tsaccver\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue\tbitscore" > $TF_BLAST
cat $BLAST_DIR/* >> $TF_BLAST
gzip $TF_BLAST
