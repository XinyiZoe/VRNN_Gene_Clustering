import torch
import pandas as pd
from src.models import VRNN
from src.train import train_vrnn
from src.utils import (
    create_vrnn_tensors,
    plotting_umap,
    save_gene_clusters_to_csv,
    plot_selected_genes,
    plot_clustered_gene_dynamics,
)

# Load data
df = pd.read_csv("./data/top_2000_most_variable_genes_RNAseq.csv")
gene_columns = df.columns[3:1003]
time_column = "Time_numeric"
cell_id_column = "PseudoCellID"

# Create tensors
data_tensor = create_vrnn_tensors(df, gene_columns, time_column, cell_id_column)

# Model setup
x_dim, h_dim, z_dim = 1, 32, 64
vrnn = VRNN(x_dim, h_dim, z_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vrnn = vrnn.to(device)
data_tensor = data_tensor.to(device)

# Train the model
vrnn, train_losses, test_losses = train_vrnn(vrnn, data_tensor=data_tensor, epochs=2000, batch_size=400, beta=1e-6, device=device)

# Run UMAP & k-clustering
embedding, cluster_labels, z_avg = plotting_umap(
    vrnn=vrnn, data_tensor=data_tensor, gene_names=list(gene_columns), device=device, n_clusters=6
)

# Plot gene dynamics by cluster (confirm valid clustering)
plot_clustered_gene_dynamics(
    vrnn=vrnn,
    data_tensor=data_tensor,
    z_latent_summary=z_avg,
    gene_names=list(gene_columns),
    df=df,
    device=device,
    n_clusters=6,
    genes_per_cluster=10,
)


# Save clusters
save_gene_clusters_to_csv(
    vrnn=vrnn,
    data_tensor=data_tensor,
    gene_names=list(gene_columns),
    device=device,
    n_clusters=6,
    output_dir="./outputs",
    file_prefix="vrnn_clusters",
)

# Plot selected genes
genes_of_interest = ["SOD2", "IL1B", "UBD", "GBP5", "CCL3"]
plot_selected_genes(
    vrnn=vrnn, data_tensor=data_tensor, gene_names=list(gene_columns), genes_of_interest=genes_of_interest, device=device
)
