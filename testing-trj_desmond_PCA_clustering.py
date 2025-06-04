#!/usr/bin/env python3
"""
pca_analysis.py

Perform PCA on C-alpha atom coordinates from MD simulation frames.

Features:
1. Plot explained variance ratio for components until 95% cumulative variance.
2. Automatically suggest optimal number of clusters using the elbow method.
3. Cluster the PCA-transformed data.
4. Plot PC1 vs PC2 and PC2 vs PC3 scatter plots with explained variance after clustering and color by cluster.
5. Compute cluster centers in PCA space.
6. Identify and save the original MD frame nearest to each cluster center as PDB files.

Usage:
    python pca_analysis.py traj_dir cms_file --output_prefix pca_results

Inputs:
    traj_dir          Desmond trajectory directory
    cms_file          Desmond .cms file (for structure and atom selection)

Options:
    --max_pc          Maximum principal components to compute (default: min(n_frames, n_calpha*3))
    --output_prefix   Prefix for saving plots and results (default: 'pca')
    --random_state    Random seed for reproducibility (default: 42)
    --asl             Atom Selection Language (ASL) string for atom selection (default: C-alphas)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from schrodinger.application.desmond.packages import analysis, topo, traj
from schrodinger.structutils import analyze
from schrodinger.structure import Structure
from schrodinger import structure



def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform PCA and clustering on MD C-alpha atoms. Pass a trajectory and .cms file"
    )
    parser.add_argument('infiles', nargs=2,
                        help='Desmond trajectory directory and corresponding .cms file')
    parser.add_argument('--max_pc', type=int, default=None,
                        help='Maximum number of principal components to compute')
    parser.add_argument('--output_prefix', type=str, default='pca',
                        help='Prefix for output files')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--asl', type=str, default='atom.ele C and atom.ptype "CA"',
                        help='ASL selection string (default: C-alpha atoms)')
    return parser.parse_args()


def read_models(cms_path: Path):
    msys_model, cms_model = topo.read_cms(str(cms_path))
    st = structure.Structure.read(str(cms_path))
    return msys_model, cms_model, st


def read_trajectory(trj_in, cms_in, asl='atom.ele C and atom.ptype "CA"'):
    msys_model, cms_model, st = read_models(Path(cms_in))
    selected = analyze.evaluate_asl(st, asl)

    frames = []
    all_frames = list(traj.read_traj(trj_in))
    for frame in all_frames:
        pos = frame.pos()
        selected_pos = pos[selected]
        frames.append(selected_pos)

    frames = np.array(frames)
    return frames, all_frames, selected, cms_model


def load_coordinates(data):
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")
    n_frames, n_calpha, _ = data.shape
    X = data.reshape(n_frames, n_calpha * 3)
    return X


def plot_pc_scatter_colored(pc_data, pca, labels, centers, output_prefix):
    var_exp = pca.explained_variance_ratio_ * 100

    plt.figure()
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(pc_data[idx, 0], pc_data[idx, 1], s=10, label=f'Cluster {lbl}')
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='k', marker='x', label='Centers')
    plt.title(f'PC1 ({var_exp[0]:.2f}%) vs PC2 ({var_exp[1]:.2f}%)')
    plt.xlabel(f'PC1 ({var_exp[0]:.2f}% variance)')
    plt.ylabel(f'PC2 ({var_exp[1]:.2f}% variance)')
    plt.legend()
    plt.savefig(f"{output_prefix}_PC1_vs_PC2.png", dpi=300)
    plt.close()

    if pc_data.shape[1] >= 3:
        plt.figure()
        for lbl in np.unique(labels):
            idx = labels == lbl
            plt.scatter(pc_data[idx, 1], pc_data[idx, 2], s=10, label=f'Cluster {lbl}')
        plt.scatter(centers[:, 1], centers[:, 2], s=100, c='k', marker='x', label='Centers')
        plt.title(f'PC2 ({var_exp[1]:.2f}%) vs PC3 ({var_exp[2]:.2f}%)')
        plt.xlabel(f'PC2 ({var_exp[1]:.2f}% variance)')
        plt.ylabel(f'PC3 ({var_exp[2]:.2f}% variance)')
        plt.legend()
        plt.savefig(f"{output_prefix}_PC2_vs_PC3.png", dpi=300)
        plt.close()


def plot_explained_variance(pca, output_prefix, variance_threshold=0.95):
    ratios = pca.explained_variance_ratio_
    cumvar = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    cutoff_idx = np.searchsorted(cumvar, variance_threshold) + 1

    plt.figure()
    plt.bar(components[:cutoff_idx], ratios[:cutoff_idx], alpha=0.7, label='Individual')
    plt.step(components[:cutoff_idx], cumvar[:cutoff_idx], where='mid', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.title('Scree Plot (up to 95% variance)')
    plt.savefig(f"{output_prefix}_explained_variance.png", dpi=300)
    plt.close()


def find_optimal_clusters(pc_data, output_prefix, max_clusters=10):
    distortions = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pc_data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(pc_data, labels))

    plt.figure()
    plt.plot(cluster_range, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (Distortion)')
    plt.title('Elbow Method For Optimal Clusters')
    plt.savefig(f"{output_prefix}_elbow.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Clusters')
    plt.savefig(f"{output_prefix}_silhouette.png", dpi=300)
    plt.close()

    best_k = cluster_range[np.argmax(silhouette_scores)]
    return best_k


def cluster_pca(pc_data, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(pc_data)
    centers = kmeans.cluster_centers_
    return labels, centers


def find_representative_frames(pc_data, centers):
    rep_frames = []
    for c in centers:
        dists = np.linalg.norm(pc_data - c, axis=1)
        rep_frames.append(int(np.argmin(dists)))
    return rep_frames


def save_representative_structures(st, trj, rep_frames, output_prefix):
    for idx, frame_idx in enumerate(rep_frames):
        new_coordinates = trj[frame_idx]
        st.setXYZ(new_coordinates)
        st.write('{}_cluster{}_rep.pdb'.format(output_prefix,idx))


def main():
    args = parse_args()

    trj_dir = args.infiles[0]
    cms_file = args.infiles[1]

    traj, all_frames, selected, cms_model = read_trajectory(trj_dir, cms_file, asl=args.asl)
    X = load_coordinates(traj)

    protein_aids = cms_model.select_atom(args.asl)
    # extract the calpha atoms into a structure
    protein_st = cms_model.extract(protein_aids)

    max_comp = args.max_pc if args.max_pc is not None else min(X.shape)
    pca = PCA(n_components=max_comp, random_state=args.random_state)
    pc_data = pca.fit_transform(X)

    plot_explained_variance(pca, args.output_prefix)

    optimal_clusters = find_optimal_clusters(pc_data, args.output_prefix)
    print(f"Optimal number of clusters suggested: {optimal_clusters}")

    labels, centers = cluster_pca(pc_data, optimal_clusters, args.random_state)

    plot_pc_scatter_colored(pc_data, pca, labels, centers, args.output_prefix)

    rep_frames = find_representative_frames(pc_data, centers)

    out_csv = Path(f"{args.output_prefix}_cluster_info.csv")
    df = pd.DataFrame({
        "Cluster Label": list(range(optimal_clusters)),
        "Center PC Coordinates": [center.tolist() for center in centers],
        "Representative Frame Index": rep_frames
    })
    df.to_csv(out_csv, index=False)

    save_representative_structures(protein_st, traj, rep_frames, args.output_prefix)

    print(f"Analysis complete. Representative frames: {rep_frames}")


if __name__ == '__main__':
    main()
