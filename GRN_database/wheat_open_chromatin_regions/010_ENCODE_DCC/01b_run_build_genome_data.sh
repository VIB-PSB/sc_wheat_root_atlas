#!/bin/bash
#$ -N build-genome
#$ -pe serial 24 
#$ -l h_vmem=10G

set -e

# This is a modified version of "build_genome_data.sh" for building the wheat genome.
# Changes are indicated with "# changed:".
# - Top lines were changed for SGE.
# - "wget" for downloading the genome file was replaced by "curl" in order to use a local file.
# - Details for the wheat genome were added (!hard coded!).
# - bowtie2-build related changes:
#   - The number of threads for bowtie2-build was increased to 24 (BUILD_BWT2_NTHREADS).
#   - The option --large-index was added (whole chr only).
#   - .bt2* was used instead of .bt2 for including .bt2l (whole chr only).
# - Removed the parts related to other genomes for shorter script.


if [[ "$#" -lt 2 ]]; then
  echo
  echo "ACTIVATE PIPELINE'S CONDA ENVIRONMENT BEFORE RUNNING THIS SCRIPT!"
  echo
  echo "This script installs data for genome [GENOME] on a directory [DEST_DIR]."
  echo "A TSV file [DEST_DIR]/[GENOME].tsv will be generated. Use it for pipeline."
  echo
  echo "Supported genomes: hg19, mm9, hg38 and mm10"
  echo
  echo "Usage: ./build_genome_data.sh [GENOME] [DEST_DIR]"
  echo "  Example: ./build_genome_data.sh hg38 /your/genome/data/path/hg38"
  echo
  exit 2
fi

# parameters for building aligner indices
BUILD_BWT2_IDX=1
BUILD_BWT2_NTHREADS=24                                                         # changed: from 2 to 24 
BUILD_BWA_IDX=0

# parameters for genome database version (v1: <ENCODE4, v3: >=ENCODE4)
VER=v3

GENOME=$1
DEST_DIR=$(cd $(dirname $2) && pwd -P)/$(basename $2)
TSV=${DEST_DIR}/${GENOME}.tsv

echo "=== Creating destination directory and TSV file..."
mkdir -p ${DEST_DIR}
cd ${DEST_DIR}


# changed: removed the parts related to other genomes
# GENOME (hg19, mm9, hg38, mm10, hg38_chr19_chrM, mm10_chr19_chrM)
# VER (v2, v3)

# triticum-aestivum_IWGSC-2.1_MtCp_split                                       # changed: wheat details
if [[ "${GENOME}" == "triticum-aestivum_IWGSC-2.1_MtCp_split" ]]; then
  # Perl style regular expression to keep regular chromosomes only.
  REGEX_BFILT_PEAK_CHR_NAME="^Chr[1-7][ABD]|^ChrUnknown"
  # mitochondrial chromosome name (e.g. chrM, MT)
  MITO_CHR_NAME="ChrMt"
  # URL for your reference FASTA (fasta, fasta.gz, fa, fa.gz, 2bit)
  REF_FA="file:///group/irc/personal/vincentp/references/triticum-aestivum_IWGSC-2.1_MtCp_split/triticum-aestivum_IWGSC-2.1_MtCp_split.fa.gz"
  # 3-col blacklist BED file to filter out overlapping peaks from b-filt peak file (.bfilt.*Peak.gz file).
  BLACKLIST=
fi


if [[ -z "${REF_FA}" ]]; then
  echo "Error: unsupported genome $GENOME"
  exit 1
fi
if [[ -z "${MITO_CHR_NAME}" ]]; then
  echo "Error: Mitochondrial chromosome name must be defined"
  exit 1
fi
if [[ -z "${REGEX_BFILT_PEAK_CHR_NAME}" ]]; then
  echo "Error: Perl style reg-ex for filtering peaks must be defined"
  exit 1
fi

echo "=== Downloading files..."
if [[ ! -z "${BLACKLIST}" ]]; then wget -N -c ${BLACKLIST}; fi
# wget -c -O $(basename ${REF_FA}) ${REF_FA}
curl -C - -o $(basename ${REF_FA}) ${REF_FA}                                   # changed: wget replacement

echo "=== Processing reference fasta file..."
if [[ ${REF_FA} == *.gz ]]; then 
  REF_FA_PREFIX=$(basename ${REF_FA} .gz)
  gzip -d -f -c ${REF_FA_PREFIX}.gz > ${REF_FA_PREFIX}
elif [[ ${REF_FA} == *.bz2 ]]; then
  REF_FA_PREFIX=$(basename ${REF_FA} .bz2)
  bzip2 -d -f -c ${REF_FA_PREFIX}.bz2 > ${REF_FA_PREFIX}
  gzip -nc ${REF_FA_PREFIX} > ${REF_FA_PREFIX}.gz
elif [[ ${REF_FA} == *.2bit ]]; then
  REF_FA_PREFIX=$(basename ${REF_FA} .2bit).fa
  twoBitToFa $(basename ${REF_FA}) ${REF_FA_PREFIX}
  gzip -nc ${REF_FA_PREFIX} > ${REF_FA_PREFIX}.gz
else
  REF_FA_PREFIX=$(basename ${REF_FA})  
fi

echo "=== Generating fasta index and chrom.sizes file..."
cd ${DEST_DIR}
samtools faidx ${REF_FA_PREFIX}
if [[ -z "${CHRSZ}" ]]; then
  CHRSZ=$GENOME.chrom.sizes
  cut -f1,2 ${REF_FA_PREFIX}.fai > ${CHRSZ}
else
  wget -N -c ${CHRSZ}
  CHRSZ="$(basename ${CHRSZ})"
fi

echo "=== Extracting mito chromosome from fasta"
REF_FA_PREFIX_WO_EXT=${REF_FA_PREFIX%.*}
REF_MITO_FA_PREFIX=${REF_FA_PREFIX_WO_EXT}.${MITO_CHR_NAME}.fa
samtools faidx ${REF_FA_PREFIX} ${MITO_CHR_NAME} > ${REF_MITO_FA_PREFIX}
gzip -nc ${REF_MITO_FA_PREFIX} > ${REF_MITO_FA_PREFIX}.gz

echo "=== Determinig gensz..."
cd ${DEST_DIR}
GENSZ=$(cat $CHRSZ | awk '{sum+=$2} END{print sum}')
if [[ "${GENOME}" == hg* ]]; then GENSZ=hs; fi
if [[ "${GENOME}" == mm* ]]; then GENSZ=mm; fi

# how to make a tar ball without permission,user,timestamp info
# https://stackoverflow.com/a/54908072

if [[ "${BUILD_BWT2_IDX}" == 1 ]]; then
  echo "=== Building bowtie2 index..."
  mkdir -p ${DEST_DIR}/bowtie2_index
  cd ${DEST_DIR}/bowtie2_index

  # whole chr
  rm -f ${REF_FA_PREFIX}
  ln -s ../${REF_FA_PREFIX} ${REF_FA_PREFIX}
  bowtie2-build ${REF_FA_PREFIX} ${REF_FA_PREFIX} --large-index --threads ${BUILD_BWT2_NTHREADS}                             # changed: added --large-index
  rm -f ${REF_FA_PREFIX}
  tar cvf ${REF_FA_PREFIX}.tar "${REF_FA_PREFIX}".*.bt2* --sort=name --owner=root:0 --group=root:0 --mtime="UTC 2019-01-01"  # changed: from .bt2 to .bt2*
  gzip -f -n ${REF_FA_PREFIX}.tar
  rm -f "${REF_FA_PREFIX}".*.bt2*                                                                                            # changed: from .bt2 to .bt2*

  # mito chr only
  rm -f ${REF_MITO_FA_PREFIX}
  ln -s ../${REF_MITO_FA_PREFIX} ${REF_MITO_FA_PREFIX}
  bowtie2-build ${REF_MITO_FA_PREFIX} ${REF_MITO_FA_PREFIX} --threads ${BUILD_BWT2_NTHREADS}
  rm -f ${REF_MITO_FA_PREFIX}
  tar cvf ${REF_MITO_FA_PREFIX}.tar "${REF_MITO_FA_PREFIX}".*.bt2 --sort=name --owner=root:0 --group=root:0 --mtime="UTC 2019-01-01"
  gzip -f -n ${REF_MITO_FA_PREFIX}.tar
  rm -f "${REF_MITO_FA_PREFIX}".*.bt2
fi

if [[ "${BUILD_BWA_IDX}" == 1 ]]; then
  echo "=== Building bwa index..."
  mkdir -p ${DEST_DIR}/bwa_index
  cd ${DEST_DIR}/bwa_index

  # whole chr
  rm -f ${REF_FA_PREFIX}
  ln -s ../${REF_FA_PREFIX} ${REF_FA_PREFIX}
  bwa index ${REF_FA_PREFIX}
  rm -f ${REF_FA_PREFIX}
  tar cvf ${REF_FA_PREFIX}.tar "${REF_FA_PREFIX}".* --sort=name --owner=root:0 --group=root:0 --mtime="UTC 2019-01-01"
  gzip -f -n ${REF_FA_PREFIX}.tar
  rm -f ${REF_FA_PREFIX}.amb ${REF_FA_PREFIX}.ann ${REF_FA_PREFIX}.bwt
  rm -f ${REF_FA_PREFIX}.pac ${REF_FA_PREFIX}.sa

  # mito chr only
  rm -f ${REF_MITO_FA_PREFIX}
  ln -s ../${REF_MITO_FA_PREFIX} ${REF_MITO_FA_PREFIX}
  bwa index ${REF_MITO_FA_PREFIX}
  rm -f ${REF_MITO_FA_PREFIX}
  tar cvf ${REF_MITO_FA_PREFIX}.tar "${REF_MITO_FA_PREFIX}".* --sort=name --owner=root:0 --group=root:0 --mtime="UTC 2019-01-01"
  gzip -f -n ${REF_MITO_FA_PREFIX}.tar
  rm -f ${REF_MITO_FA_PREFIX}.amb ${REF_MITO_FA_PREFIX}.ann ${REF_MITO_FA_PREFIX}.bwt
  rm -f ${REF_MITO_FA_PREFIX}.pac ${REF_MITO_FA_PREFIX}.sa
fi

echo "=== Removing temporary files..."
cd ${DEST_DIR}
rm -f ${REF_FA_PREFIX} ${REF_MITO_FA_PREFIX}

echo "=== Creating TSV file... (${TSV})"
cd ${DEST_DIR}
rm -f ${TSV}
touch ${TSV}

echo -e "genome_name\t${GENOME}" >> ${TSV}
echo -e "ref_fa\t${DEST_DIR}/${REF_FA_PREFIX}.gz" >> ${TSV}
echo -e "ref_mito_fa\t${DEST_DIR}/${REF_MITO_FA_PREFIX}.gz" >> ${TSV}
echo -e "mito_chr_name\t${MITO_CHR_NAME}" >> ${TSV}
printf "regex_bfilt_peak_chr_name\t%s\n" "${REGEX_BFILT_PEAK_CHR_NAME}" >> ${TSV}
if [[ ! -z "${BLACKLIST}" ]]; then
  echo -e "blacklist\t${DEST_DIR}/$(basename ${BLACKLIST})" >> ${TSV};
fi
echo -e "chrsz\t${DEST_DIR}/$(basename ${CHRSZ})" >> ${TSV}
echo -e "gensz\t${GENSZ}" >> ${TSV}
if [[ ${BUILD_BWT2_IDX} == 1 ]]; then
  echo -e "bowtie2_idx_tar\t${DEST_DIR}/bowtie2_index/${REF_FA_PREFIX}.tar.gz" >> ${TSV}
  echo -e "bowtie2_mito_idx_tar\t${DEST_DIR}/bowtie2_index/${REF_MITO_FA_PREFIX}.tar.gz" >> ${TSV}
fi
if [[ ${BUILD_BWA_IDX} == 1 ]]; then
  echo -e "bwa_idx_tar\t${DEST_DIR}/bwa_index/${REF_FA_PREFIX}.tar.gz" >> ${TSV}
  echo -e "bwa_mito_idx_tar\t${DEST_DIR}/bwa_index/${REF_MITO_FA_PREFIX}.tar.gz" >> ${TSV}
fi

echo "=== Downloading ATAQC file... (${TSV})"
cd ${DEST_DIR}
mkdir -p ataqc
cd ataqc

if [[ ! -z "${TSS}" ]]; then
  wget -N -c ${TSS}
  echo -e "tss\t${DEST_DIR}/ataqc/$(basename ${TSS})" >> ${TSV}
fi
if [[ ! -z "${DNASE}" ]]; then
  wget -N -c ${DNASE}
  echo -e "dnase\t${DEST_DIR}/ataqc/$(basename ${DNASE})" >> ${TSV}
fi
if [[ ! -z "${PROM}" ]]; then
  wget -N -c ${PROM}
  echo -e "prom\t${DEST_DIR}/ataqc/$(basename ${PROM})" >> ${TSV}
fi
if [[ ! -z "${ENH}" ]]; then
  wget -N -c ${ENH}
  echo -e "enh\t${DEST_DIR}/ataqc/$(basename ${ENH})" >> ${TSV}
fi
if [[ ! -z "${REG2MAP}" ]]; then
  wget -N -c ${REG2MAP}
  echo -e "reg2map\t${DEST_DIR}/ataqc/$(basename ${REG2MAP})" >> ${TSV}
fi
if [[ ! -z "${REG2MAP_BED}" ]]; then
  wget -N -c ${REG2MAP_BED}
  echo -e "reg2map_bed\t${DEST_DIR}/ataqc/$(basename ${REG2MAP_BED})" >> ${TSV}
fi
if [[ ! -z "${ROADMAP_META}" ]]; then
  wget -N -c ${ROADMAP_META}
  echo -e "roadmap_meta\t${DEST_DIR}/ataqc/$(basename ${ROADMAP_META})" >> ${TSV}
fi

echo "=== All done."
