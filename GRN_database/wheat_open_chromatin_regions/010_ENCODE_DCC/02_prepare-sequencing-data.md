# Prepare sequencing data

A set of commands to launch manually for downloading, preparing and checking sequencing data.

********************************************************************************

# Variables

```bash
# Define directories
DIR_MAIN=/group/irc/personal/vincentp

DIR_RAW=$DIR_MAIN/raw/public-ATACseq_wang-2022
SET=wang-2022_AK58_spike_dr

# Define files (SRA or URL, name)
LINKS=(
SRR16929302,pei-2022_AK58_root_2wo_rep1_SRR16929302
SRR16929303,pei-2022_AK58_root_2wo_rep2_SRR16929303
)
```

********************************************************************************

# Prepare sequences

## From SRA

**prepare_sra.bash**:
```bash
#!/bin/bash
#$ -N download
#$ -cwd
#$ -pe serial 24
#$ -l h_vmem=10G

module load sra_toolkit/x86_64/3.0.0
prefetch -v -p $SRA -O .
fasterq-dump -v -p -e 24 $SRA -o ./$NAME
rm -r $SRA
```

```bash
cd $DIR_RAW

while IFS=$',' read -r col1 col2; do
  SRA=$(echo $col1)
  NAME=$(echo $col2)
  qsub -b y -N download -l h_vmem=10G \
    qsub prepare_sra.bash
done <<< "$LINKS"
```

Once the whole set is downloaded:

```bash
cd $DIR_RAW
ls -lh | grep $SET > $SET\_size
for file in $SET*; do
  qsub -b y -l h_vmem=10G gzip $file
done
```

## From URL

```bash
cd $DIR_RAW

while IFS=$',' read -r col1 col2; do
  URL=$(echo $col1)
  NAME=$(echo $col2)
  qsub -b y -N download -l h_vmem=10G \
    wget -P $DIR_RAW -O $NAME $URL
done <<< "$LINKS"
```

********************************************************************************

# Quality check

## FastQC

```bash
cd $DIR_RAW
mkdir $SET\_fastqc

module load FastQC
for file in $SET*gz; do
  qsub -b y -V -cwd -l h_vmem=10G -N fastQC \
    fastqc --noextract --outdir $SET\_fastqc $file
done
```

## MultiQC

```bash
cd $DIR_RAW/$SET\_fastqc

module load multiqc
qsub -b y -V -cwd -l h_vmem=10G multiqc .
```
