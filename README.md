# VRNN_Gene_Clustering for Single-Cell ATAC-Seq

This repository implements a Variational Recurrent Neural Network (VRNN) to model temporal gene activity dynamics from single-cell ATAC-seq data. The learned latent space is used to cluster genes based on their response profiles following stimulation (e.g., IFNγ). This enables the identification of gene modules—such as fast responders (e.g., IRF1) versus delayed responders (e.g., CXCL9/10)—and provides a framework for downstream inference of cis-regulatory elements (CREs) that may control these differences.

## Why It Matters  

Immune response genes often display distinct temporal patterns: some activate within minutes, while others turn on hours later. We hypothesize that these differences are regulated by distinct cis-regulatory elements, which coordinate timing and magnitude of gene expression. By applying a VRNN to scATAC-seq data, we capture latent temporal dynamics, cluster genes by behavior, and highlight potential regulatory modules. This pipeline demonstrates how deep generative models can guide biological hypothesis generation, linking computational predictions to mechanisms of immune activation.  

---

## Quick Start

Clone the repository, install dependencies, and run the pipeline on the included sample dataset:

```bash
# 1. Clone the repo
git clone https://github.com/XinyiZoe/VRNN_Gene_Clustering.git
cd VRNN_Gene_Clustering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model on the sample subset (53 MB from GSE164498)
python main.py --input data/sample_GSE164498_subset.csv --output results/demo.png
```

## Project Goals  

This project applies deep generative modeling to uncover regulatory insights from single-cell time-series data. Specifically, we aim to:  

- **Model temporal dynamics** of gene activity using a Variational Recurrent Neural Network (VRNN), capturing how accessibility changes in response to stimulation.  
- **Cluster genes by latent dynamics**, revealing modules of fast vs. slow responders (e.g., IRF1 vs. CXCL9/10).
- **Identify co-regulated modules** of genes with similar activity profiles, providing candidates for follow-up cis-regulatory element (CRE) analysis.
- **Bridge computational outputs with biology** by linking clusters to immune activation pathways and regulatory mechanisms.  

**Example outputs**L clustering the 2000 most variable genes with k = 5.

![Clustering trajectories (from randomly selected genes per cluster)](results/figures/2000_genes_k=5.png)
![Clustering results(overall range)](results/figures/2000_genes_k=5_range.png)

## Data

This project uses **GSE164498 scATAC-seq** data from GEO.

- `sample_GSE164498_subset.csv` (53 MB): the top 2000 most variable genes from m1 24 hours data. 
- Full dataset: [GSE164498 on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164498)

To download the full dataset:
```bash
bash scripts/download_data.sh
```

## Directory Structure

```
vrnn-clustering/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/                  # Sample dataset (subset of GSE164498)
├── results/               # Example plots/figures
├── scripts/               # Utility scripts (e.g., download_data.sh)
└── src/
    ├── models.py
    ├── train.py
    └── utils.py
```

## Installation
```
pip install -r requirements.txt
```

## Input Data Format

The ATAC-seq dataset is structured with metadata columns (Cell_ID, CellType, Time_numerics) followed by a gene activity (accessibility) matrix, where each column represents a gene and each entry reflects the accessibility score inferred for that gene in a given cell.

By default, the model uses the first 1000 genes after column index 3. You can modify this in main.py.
Make sure to sort the cells by time nuumerics (earliest to latest) as this model assumes cells are ordered by time. 

## Core Modules
``` 
main.py
```
- Orchestrates the pipeline.
- Loads data, trains the model, extracts latent features, performs clustering, and generates visualizations.

```
models.py
```
Implement the VRNN architecture:
- GRU-based recurrence.
- Gaussian prior/posterior for latent z.
- Reconstruction and KL loss terms.

```
train.py
```
- Manages training and validation loops.
- Handles batching, optimization, logging of loss curves.
- Model checkpoints can be added here if needed.

```
utils.py
```
include methods that: 
- create VRNN Cluster
- plot the umap model based on latent clusters post-training
- plot gene dynamics (default: gene_per_cluster = 10) to ensure appropriate clusters are formed
- save each cluster to .csv format
- plot selected genes based on gene names
  
## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



