#!/usr/bin/env python3
"""
pca_analysis.py

Perform PCA on C-alpha atom coordinates from MD simulation frames.

Features:
1. Plot PC1 vs PC2 and PC2 vs PC3 scatter plots.
2. Plot explained variance ratio for all components (scree plot) until cumulative variance reaches 100%.
3. Cluster the PCA-transformed data in user-specified number of clusters.
4. Compute cluster centers in PCA space.
5. Identify the original MD frame nearest to each cluster center.

Usage:
    python pca_analysis.py coords.npy --n_clusters 4 --output_prefix pca_results

Inputs:
    coords.npy        NumPy file containing array of shape (n_frames, n_calpha, 3)

Options:
    --n_clusters      Number of clusters for KMeans (default: 4)
    --max_pc          Maximum principal components to compute (default: min(n_frames, n_calpha*3))
    --output_prefix   Prefix for saving plots and results (default: 'pca')
    --random_state    Random seed for reproducibility (default: 42)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
from schrodinger.application.desmond.packages import analysis, topo, traj
from schrodinger import structure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform PCA and clustering on MD C-alpha atoms. Pass a concatenated trj"
    )
    # Input trajectory directories (one or more)
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of clusters for KMeans')
    parser.add_argument('--max_pc', type=int, default=None,
                        help='Maximum number of principal components to compute')
    parser.add_argument('--output_prefix', type=str, default='pca',
                        help='Prefix for output files')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def read_traj(trj_path: Path) -> list:
    """
    Read trajectory frames from a Desmond directory.
    """
    # Convert generator to list for multiple passes
    return list(traj.read_traj(str(trj_path)))


def read_models(cms_path: Path):
    """
    Load MSYS and CMS models, plus structure for reference.
    """
    # Parse CMS file to get MSYS and CMS models
    msys_model, cms_model = topo.read_cms(str(cms_path))
    # Read structure for ASL evaluation
    st = structure.Structure.read(str(cms_path))
    return msys_model, cms_model, st


def load_coordinates(trj):
    """
    Load coordinates from a Desmond trajectory.
    Returns:
        X: ndarray of shape (n_frames, n_calpha*3)
    """
    # get the frame coordinates for euclidean (xyz) distance measurement
    frame_coords = [f.pos() for f in trj]
    return frame_coords


def plot_pc_scatter(pc_data, output_prefix):
    """
    Plot PC1 vs PC2 and PC2 vs PC3 scatter plots.
    """
    plt.figure()
    plt.scatter(pc_data[:, 0], pc_data[:, 1], s=10)
    plt.title('PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f"{output_prefix}_PC1_vs_PC2.png", dpi=300)
    plt.close()

    if pc_data.shape[1] >= 3:
        plt.figure()
        plt.scatter(pc_data[:, 1], pc_data[:, 2], s=10)
        plt.title('PC2 vs PC3')
        plt.xlabel('PC2')
        plt.ylabel('PC3')
        plt.savefig(f"{output_prefix}_PC2_vs_PC3.png", dpi=300)
        plt.close()


def plot_explained_variance(pca, output_prefix):
    """
    Plot explained variance ratio and cumulative variance.
    """
    ratios = pca.explained_variance_ratio_
    cumvar = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    plt.figure()
    plt.bar(components, ratios, alpha=0.7, label='Individual')
    plt.step(components, cumvar, where='mid', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.title('Scree Plot')
    plt.savefig(f"{output_prefix}_explained_variance.png", dpi=300)
    plt.close()


def cluster_pca(pc_data, n_clusters, random_state):
    """
    Cluster PCA data using KMeans and return labels and centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(pc_data)
    centers = kmeans.cluster_centers_
    return labels, centers


def find_representative_frames(pc_data, centers):
    """
    For each center, find the index of the frame closest to it.
    """
    rep_frames = []
    for c in centers:
        # Euclidean distance
        dists = np.linalg.norm(pc_data - c, axis=1)
        rep_frames.append(int(np.argmin(dists)))
    return rep_frames


def save_cluster_scatter(pc_data, labels, centers, output_prefix):
    """
    Plot clustered PCA with cluster centers.
    """
    plt.figure()
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(pc_data[idx, 0], pc_data[idx, 1], s=10, label=f'Cluster {lbl}')
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='k', marker='x', label='Centers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA Clusters')
    plt.savefig(f"{output_prefix}_clusters.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    # Load and prepare data
    X = load_coordinates(args.infiles)

    # TODO: need to be able to handle more than 1 trj

    # Determine max components
    max_comp = args.max_pc or min(X.shape)
    pca = PCA(n_components=max_comp, random_state=args.random_state)
    pc_data = pca.fit_transform(X)

    # Plot PCA scatter and variance
    plot_pc_scatter(pc_data, args.output_prefix)
    plot_explained_variance(pca, args.output_prefix)

    # Cluster in PCA space
    labels, centers = cluster_pca(pc_data, args.n_clusters, args.random_state)
    save_cluster_scatter(pc_data, labels, centers, args.output_prefix)

    # Find representative frames for each cluster
    rep_frames = find_representative_frames(pc_data, centers)

    # Save cluster information
    out_txt = Path(f"{args.output_prefix}_cluster_info.txt")
    with out_txt.open('w') as f:
        f.write('Cluster Label, Center PC Coordinates, Representative Frame Index\n')
        for lbl, center, frame in zip(range(args.n_clusters), centers, rep_frames):
            f.write(f"{lbl}, {center.tolist()}, {frame}\n")

    print(f"Analysis complete. Representative frames: {rep_frames}")

if __name__ == '__main__':
    main()