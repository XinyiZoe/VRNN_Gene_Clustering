# src/utils.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import umap


def create_vrnn_tensors(df, genes, time_col, cell_id_col, approach='mean'):
    if approach == 'mean':
        df_mean = df.groupby(time_col)[genes].mean()
        tensor = torch.tensor(df_mean.T.values, dtype=torch.float32).unsqueeze(-1)
        df_mean.to_csv('./VRNN_mean_expression.csv')
    print("Data tensor shape:", tensor.shape)
    return tensor


def plotting_umap(
    vrnn,
    data_tensor,
    gene_names,
    device,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    add_gene_labels=True,
    apply_kmeans=True,
    n_clusters=3,
    random_state=42,
):
    vrnn.eval()
    with torch.no_grad():
        outputs = vrnn(data_tensor.to(device))
        z_samples = outputs["z"].cpu().numpy()
    z_avg = z_samples.mean(axis=1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(z_avg)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(z_avg) if apply_kmeans else None

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=cluster_labels.astype(str) if apply_kmeans else None, palette="Set1", s=80
    )
    if add_gene_labels:
        for i, gene in enumerate(gene_names):
            plt.text(embedding[i, 0], embedding[i, 1], gene, fontsize=8, ha='right', va='top')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP Projection of VRNN Latent Space")
    plt.tight_layout()
    plt.show()
    return embedding, cluster_labels, z_avg


def plot_clustered_gene_dynamics(
    vrnn,
    data_tensor,
    z_latent_summary,
    gene_names,
    df,
    device,
    time_column="Time_numeric",
    n_clusters=2,
    genes_per_cluster=10,
    random_state=42,
):
    vrnn.eval()
    with torch.no_grad():
        outputs = vrnn(data_tensor.to(device))
        x_mean = outputs["x_mean"].cpu().numpy().squeeze(-1)
    timepoints = sorted(df[time_column].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(z_latent_summary)
    cluster_gene_indices = [
        np.random.choice(np.where(cluster_labels == i)[0], size=min(genes_per_cluster, sum(cluster_labels == i)), replace=False)
        for i in range(n_clusters)
    ]
    fig, axs = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axs = [axs]
    for cluster_id, gene_indices in enumerate(cluster_gene_indices):
        ax = axs[cluster_id]
        for idx in gene_indices:
            ax.plot(timepoints, x_mean[idx], label=gene_names[idx], linewidth=2)
        ax.set_title(f"Cluster {cluster_id + 1}")
        ax.set_xlabel("Time")
        if cluster_id == 0:
            ax.set_ylabel("Expression")
        ax.legend(fontsize=8)
    plt.suptitle(f"VRNN-Reconstructed Gene Dynamics\n({genes_per_cluster} Genes per Cluster, KMeans k={n_clusters})")
    plt.tight_layout()
    plt.show()


def save_gene_clusters_to_csv(
    vrnn,
    data_tensor,
    gene_names,
    device,
    n_clusters=3,
    output_dir="./",
    file_prefix="gene_clusters",
    split_files=False,
    random_state=42,
):
    vrnn.eval()
    with torch.no_grad():
        outputs = vrnn(data_tensor.to(device))
        z_samples = outputs["z"].cpu().numpy()
    z_avg = z_samples.mean(axis=1)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(z_avg)
    df_clusters = pd.DataFrame({"Gene": gene_names, "Cluster": cluster_labels})
    os.makedirs(output_dir, exist_ok=True)
    if split_files:
        for i in range(n_clusters):
            df_sub = df_clusters[df_clusters["Cluster"] == i]
            df_sub.to_csv(os.path.join(output_dir, f"{file_prefix}_cluster_{i + 1}.csv"), index=False)
    else:
        df_clusters.to_csv(os.path.join(output_dir, f"{file_prefix}_k{n_clusters}.csv"), index=False)
    return df_clusters


def plot_selected_genes(vrnn, data_tensor, gene_names, genes_of_interest, device):
    vrnn.eval()
    with torch.no_grad():
        outputs = vrnn(data_tensor.to(device))
        x_mean = outputs["x_mean"].cpu().numpy().squeeze(-1)
    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    indices = [gene_idx_map[g] for g in genes_of_interest if g in gene_idx_map]
    plt.figure(figsize=(10, 6))
    for g, i in zip(genes_of_interest, indices):
        plt.plot(range(x_mean.shape[1]), x_mean[i], label=g, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Mean Expression (VRNN Output)")
    plt.title("Selected Gene Expression Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
