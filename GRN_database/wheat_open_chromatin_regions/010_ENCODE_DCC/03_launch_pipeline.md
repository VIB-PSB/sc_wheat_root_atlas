Run ENCODE-DCC atac-seq-pipeline
================================================================================

A set of manual commands for launching the ENCODE-DCC atac-seq pipeline.
This will submit many (large) jobs to the cluster.

Pipelines links:

- Overview: https://www.encodeproject.org/atac-seq
- GitHub: https://github.com/ENCODE-DCC/atac-seq-pipeline

********************************************************************************

# Setup

- Configure default.conf and potentially atac.wdl (requires testing)
- Prepare sequencing data (.fastq.gz)
- Configure the corresponding .json file

********************************************************************************

# Process

## 1. Add the correct modules as autoload

```bash
module autoload append java/x86_64/16.0.1+9
module autoload append python/x86_64/3.8.0
```

## 2. Modify the input JSON file

See [here](https://github.com/ENCODE-DCC/atac-seq-pipeline#input-json-file-specification).

## 3. Run pipeline

```bash
module load caper
caper hpc submit atac.wdl -i "${INPUT_JSON}" --singularity --leader-job-name NAME --db file --file-db ./run_1/run_1
```

Notes:
- It is better to launch from the working directory (not the pipeline one).
- It takes a little while to start.
- To be able to resume from the cache, set `--db` to `file` and `--file-db` to a specific path. Use the exact same command line to resume.
- To cancel, use `caper hpc list` to list all leader jobs. Use `caper hpc abort JOB_ID` to abort a running leader job. Do not directly cancel a job using cluster command like `scancel` or `qdel`, as only the leader job will be canceled.

## 4. Organize output and make a spreadsheet of QC metrics

```bash
# croo and qc2tsv were installed within python/x86_64/3.8.0

# croo
module load python/x86_64/3.8.0
module load graphviz
METADATA_JSON_FILE=$(find -name "metadata.json")
mkdir croo
cd croo
qsub -b y -V -cwd -l h_vmem=10G croo ../$METADATA_JSON_FILE

# qc2tsv (within an interactive session for now)
qsub -b y -V -cwd -l h_vmem=10G qc2tsv qc/qc.json > qc2tsv.tsv
```
## 5. Export

Export all results from the `croo` folder except BAM files.
