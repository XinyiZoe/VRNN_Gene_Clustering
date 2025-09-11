# VRNN Gene Clustering for Single-Cell ATAC-Seq

This repository implements a Variational Recurrent Neural Network (VRNN) to model temporal gene activity dynamics from single-cell ATAC-seq data. The learned latent space is used to cluster genes based on their response profiles following stimulation (e.g., IFNγ). This enables the identification of gene modules—such as fast responders (e.g., IRF1) versus delayed responders (e.g., CXCL9/10)—and provides a framework for downstream inference of cis-regulatory elements (CREs) that may control these differences.

## Why It Matters  

Immune response genes often display distinct temporal patterns: some activate rapidly within minutes, while others turn on hours later. For example, **IRF1 is a fast responder to IFNγ stimulation, while CXCL9 and CXCL10 are slower and more heterogeneous in their activation trajectories** ([Naigles et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37689116/)). These differences suggest regulation by distinct cis-regulatory elements (CREs), which coordinate both the timing and magnitude of gene expression.  

By applying a VRNN to scATAC-seq data, this project captures latent temporal dynamics, clusters genes by behavior, and highlights potential regulatory modules. In doing so, it demonstrates how deep generative models can generate biologically testable hypotheses about CREs and their roles in immune activation.  

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

**Example outputs:** Clustering of the 2000 most variable genes (*k* = 5).  

- **Figure 1A**: Trajectories of randomly selected genes from each cluster, illustrating representative dynamic patterns.  
- **Figure 1B**: Summary of all clusters, showing the maximum and minimum trajectories within each group to highlight overall range and variability.  

**Figure 1A:**  
![Clustering trajectories (randomly selected genes per cluster)](results/figures/2000_genes_k=5.png)  

**Figure 1B:**  
![Clustering results (overall range)](results/figures/2000_genes_k=5_range.png)  


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

The ATAC-seq dataset is structured with metadata columns (`Cell_ID`, `CellType`, `Time_numerics`) followed by a gene activity (accessibility) matrix.  
Each column represents a gene, and each entry reflects the accessibility score inferred for that gene in a given cell.  

- By default, the model uses the first 1000 genes after column index 3 (modifiable in `main.py`).  
- Cells must be sorted by `Time_numerics` (earliest to latest), as the VRNN assumes chronological ordering.  

```{python, eval=FALSE}
import pandas as pd

# Example of expected structure
data = pd.DataFrame({
    "Cell_ID": ["cell_1", "cell_2", "cell_3"],
    "CellType": ["macrophage", "macrophage", "macrophage"],
    "Time_numerics": [0, 1, 2],
    "GeneA": [0.1, 0.3, 0.5],
    "GeneB": [0.0, 0.2, 0.7],
    "GeneC": [0.05, 0.15, 0.25]
})

print(data.head())
```

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
Provide helper methods that:
- Create VRNN Cluster
- Plot UMAP embeddings of latent clusters
- Plot gene dynamics (default = 10 per cluster)
- Export each cluster to .csv format
- Visualize selected genes based on gene names
  
## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



