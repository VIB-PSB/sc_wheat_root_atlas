#!/bin/bash
#$ -N FIMO_MapMotifs
#$ -cwd
#$ -l h_vmem=10G
set -e

# Prepare environment
while [ $# -gt 0 ]; do
    case "$1" in
        --motifs)
            MOTIFS="$2"
            shift 2
            ;;
        --dir_motifs)
            DIR_MOTIFS="$2"
            shift 2
            ;;
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --dir_out)
            DIR_OUT="$2"
            shift 2
            ;;
        *)
			break
            ;;
    esac
done

module load meme

# Run fimo for each motif
for MOTIF in $MOTIFS; do
  NAME=${MOTIF/.txt/}
  fimo --oc $DIR_OUT/$NAME --max-stored-scores 1000000 $DIR_MOTIFS/$MOTIF $SEQUENCES
  gzip -f $DIR_OUT/$NAME/fimo.tsv
  # Remove unnecessary created files
  rm $DIR_OUT/$NAME/cisml.xml $DIR_OUT/$NAME/fimo.gff $DIR_OUT/$NAME/fimo.html $DIR_OUT/$NAME/fimo.xml
done

echo "Script finished ($(date))."
