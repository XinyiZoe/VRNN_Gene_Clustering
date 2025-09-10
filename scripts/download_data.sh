#!/bin/bash
set -euo pipefail

# Directory to store full data
DATA_DIR="data/full_GSE164498"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading processed ATAC-seq supplementary files for GSE164498 …"

# GEO supplementary files (bed files, tables, raw tarball)
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_HL60_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M1_12hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M1_24hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M1_3hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M1_6hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_12hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_24hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_3hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_6hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_ID2_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_IRF1_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_IRF7_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_IRF9_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_Neg_Ctrl_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_M2_Pos_Ctrl_KD_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_Mac_0hrs_bATACseq_IDRoutput.bed.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164498/suppl/GSE164498_RAW.tar

echo "Download complete. Data saved in $DATA_DIR"

