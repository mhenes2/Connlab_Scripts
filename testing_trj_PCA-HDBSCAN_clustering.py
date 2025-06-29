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
    --min_cluster_size  Minimum cluster size for HDBSCAN (default: 10)
    --min_samples       Minimum samples for HDBSCAN (default: same as min_cluster_size)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
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
    parser.add_argument('--min_cluster_size', type=int, default=10,
                        help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--min_samples', type=int, default=None,
                        help='Minimum samples for HDBSCAN (defaults to min_cluster_size)')
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
        frames.append(pos[selected])
    frames = np.array(frames)
    return frames, all_frames, selected, cms_model


def load_coordinates(data):
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")
    n_frames, n_calpha, _ = data.shape
    return data.reshape(n_frames, n_calpha * 3)


def plot_pc_scatter_colored(pc_data, pca, labels, centers, output_prefix):
    var_exp = pca.explained_variance_ratio_ * 100
    plt.figure()
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(pc_data[idx,0], pc_data[idx,1], s=10, label=f'Cluster {lbl}')
    plt.scatter(centers[:,0], centers[:,1], s=100, c='k', marker='x', label='Centers')
    plt.title(f'PC1 ({var_exp[0]:.2f}%) vs PC2 ({var_exp[1]:.2f}%)')
    plt.xlabel(f'PC1 ({var_exp[0]:.2f}% var)')
    plt.ylabel(f'PC2 ({var_exp[1]:.2f}% var)')
    plt.legend()
    plt.savefig(f"{output_prefix}_PC1_vs_PC2.png", dpi=300)
    plt.close()
    if pc_data.shape[1] >=3:
        plt.figure()
        for lbl in np.unique(labels):
            idx = labels==lbl
            plt.scatter(pc_data[idx,1], pc_data[idx,2], s=10, label=f'Cluster {lbl}')
        plt.scatter(centers[:,1], centers[:,2], s=100, c='k', marker='x', label='Centers')
        plt.title(f'PC2 ({var_exp[1]:.2f}%) vs PC3 ({var_exp[2]:.2f}%)')
        plt.xlabel(f'PC2 ({var_exp[1]:.2f}% var)')
        plt.ylabel(f'PC3 ({var_exp[2]:.2f}% var)')
        plt.legend()
        plt.savefig(f"{output_prefix}_PC2_vs_PC3.png", dpi=300)
        plt.close()


def plot_explained_variance(pca, output_prefix, variance_threshold=0.95):
    ratios = pca.explained_variance_ratio_
    cumvar = np.cumsum(ratios)
    cutoff = np.searchsorted(cumvar, variance_threshold)+1
    comps = np.arange(1,cutoff+1)
    plt.figure()
    plt.bar(comps, ratios[:cutoff], alpha=0.7)
    plt.step(comps, cumvar[:cutoff], where='mid')
    plt.xlabel('PC'); plt.ylabel('Variance Explained')
    plt.title('Scree Plot (95%)')
    plt.savefig(f"{output_prefix}_explained_variance.png", dpi=300)
    plt.close()


def find_optimal_clusters(pc_data, output_prefix, max_clusters=10):
    distortions=[]; sils=[]
    ks=range(2,max_clusters+1)
    for k in ks:
        km=KMeans(n_clusters=k,random_state=42).fit(pc_data)
        lab=km.labels_
        distortions.append(km.inertia_)
        sils.append(silhouette_score(pc_data,lab))
    # might plot if desired
    return ks[np.argmax(sils)]


def cluster_kmeans(pc_data, n_clusters, random_state):
    km=KMeans(n_clusters=n_clusters, random_state=random_state)
    labs=km.fit_predict(pc_data)
    return labs, km.cluster_centers_


def cluster_hdbscan(pc_data, min_cluster_size, min_samples=None):
    if min_samples is None: min_samples=min_cluster_size
    clstr=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labs=clstr.fit_predict(pc_data)
    labs_ids=sorted([l for l in set(labs) if l!=-1])
    ctrs=np.vstack([pc_data[labs==l].mean(axis=0) for l in labs_ids]) if labs_ids else np.empty((0,pc_data.shape[1]))
    return labs, ctrs, labs_ids

# New function to find optimal HDBSCAN cluster size via silhouette
def find_optimal_hdbscan(pc_data, cluster_sizes, min_samples=None):
    """
    Determine optimal min_cluster_size for HDBSCAN by maximizing silhouette score.
+    Args:
+        pc_data (ndarray): PCA-transformed data.
+        cluster_sizes (list[int]): Values to test for min_cluster_size.
+        min_samples (int): If None, uses same as min_cluster_size.
+    Returns:
+        best_size (int), best_labels (ndarray), best_centers (ndarray), best_ids (list)
+    """
    best_score=-1.0
    best_out=(None,None,None,None)
    for size in cluster_sizes:
        labs, ctrs, ids = cluster_hdbscan(pc_data, size, min_samples)
        # require at least 2 clusters for silhouette
        if len(ids) > 1:
            mask = labs != -1
            try:
                score = silhouette_score(pc_data[mask], labs[mask])
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_out = (size, labs, ctrs, ids)
    return best_out


def find_representative_frames(pc_data, centers):
    reps=[]
    for c in centers:
        d=np.linalg.norm(pc_data-c,axis=1)
        reps.append(int(np.argmin(d)))
    return reps


def save_representative_structures(st, frames, rep_frames, output_prefix):
    for i,idx in enumerate(rep_frames):
        st.setXYZ(frames[idx])
        st.write(f"{output_prefix}_cluster{i}.pdb")


def main():
    args=parse_args()
    trj,cms=args.infiles
    data, all_frames, sel, cms_model=read_trajectory(trj,cms,args.asl)
    X=load_coordinates(data)
    aids=cms_model.select_atom(args.asl)
    st_elem=cms_model.extract(aids)
    maxc=args.max_pc or min(X.shape)
    pca=PCA(n_components=maxc,random_state=args.random_state)
    pcd=pca.fit_transform(X)
    plot_explained_variance(pca,args.output_prefix)
    k=find_optimal_clusters(pcd,args.output_prefix)
    print(f"Optimal KMeans clusters: {k}")
    kml,kmc=cluster_kmeans(pcd,k,args.random_state)
    plot_pc_scatter_colored(pcd,pca,kml,kmc,args.output_prefix+"_kmeans")
    kmreps=find_representative_frames(pcd,kmc)
    pd.DataFrame({'Cluster':list(range(k)),'Center':[c.tolist() for c in kmc],'Frame':kmreps}).to_csv(f"{args.output_prefix}_kmeans.csv",index=False)
    # save_representative_structures(st_elem, all_frames, kmreps, args.output_prefix+"_kmeans")
    # test multiple sizes
    sizes=range(5, args.min_cluster_size*3, args.min_cluster_size)
    best_size, hdl, hdc, hids = find_optimal_hdbscan(pcd, list(sizes), args.min_samples)
    print(f"Optimal HDBSCAN min_cluster_size: {best_size}")
    plot_pc_scatter_colored(pcd,pca,hdl,hdc,args.output_prefix+"_hdbscan")
    hdreps=find_representative_frames(pcd,hdc)
    pd.DataFrame({'Cluster':hids,'Center':[c.tolist() for c in hdc],'Frame':hdreps}).to_csv(f"{args.output_prefix}_hdbscan.csv",index=False)
    # save_representative_structures(st_elem, all_frames, hdreps, args.output_prefix+"_hdbscan")

if __name__=='__main__':
    main()
