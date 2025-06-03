#!/usr/bin/env python3
"""
gpu_distance_auto.py

This script calculates intraprotein Cα distances from one or more Desmond trajectories.
It always writes out three CSV files:
  1) A square (n_atoms×n_atoms) “average” distance matrix with row/column labels.
  2) A “per‐replicate + average” CSV where each row is one ASL‐selected atom‐pair and columns are:
       Pair, Rep1, Rep2, …, RepN, Average
  3) A “average‐only” CSV where each row is one ASL‐selected atom‐pair and columns are:
       Pair, Average

It automatically picks the GPU with the most free memory rather than defaulting to GPU0,
and uses CuPy to batch‐compute distances on GPU.

Usage:
    python gpu_distance_auto.py traj_dir1 [traj_dir2 ...] \
        -cms_file /path/to/model.out.cms \
        -outname output_prefix \
        [-N 0.5] [-asl "protein and a. CA"] [-s "0:-1:1"]

Dependencies:
  • pynvml                (pip install pynvml)
  • schrodinger (traj, topo)
  • numpy, scipy, psutil, argparse, csv
  • cupy (CUDA-enabled)
"""

import os
import sys

# ─────────── 1) Auto‐select GPU via pynvml and set CUDA_VISIBLE_DEVICES ───────────
try:
    import pynvml
except ImportError:
    sys.exit("ERROR: pynvml not found. Please install via `pip install pynvml` and try again.")

pynvml.nvmlInit()
gpu_count = pynvml.nvmlDeviceGetCount()
selected_gpu = None
max_free_mem = 0

for i in range(gpu_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mem = info.free
    if free_mem > max_free_mem:
        max_free_mem = free_mem
        selected_gpu = i

if selected_gpu is None:
    sys.exit("No CUDA GPU available—exiting.")

# Tell ALL CUDA libraries (including CuPy) to see only our chosen GPU as “cuda:0”
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
print(f"[*] Selected physical GPU {selected_gpu} "
      f"(free ≈ {max_free_mem/1e9:.2f} GB). "
      f"Setting CUDA_VISIBLE_DEVICES={selected_gpu}")

# ─────────── 2) Now import the rest of the modules (after setting CUDA_VISIBLE_DEVICES) ───────────
import argparse
import csv
from datetime import datetime
import psutil

import numpy as np
from scipy.spatial.distance import euclidean

from schrodinger.application.desmond.packages import traj, topo

import cupy as cp  # Now “cuda:0” refers to whichever GPU we picked above

# Record script start time
startTime = datetime.now()


def dynamic_cpu_assignment(fraction):
    """
    Use a fraction of the available CPUs.
    :param fraction: If >= 1, treated as exact # of CPUs.
                     If < 1, fraction of currently available CPUs.
    :return: int (# of worker processes to spawn)
    """
    if fraction >= 1:
        return int(fraction)
    ncpus = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=2)
    avail_cpus = int(ncpus - (ncpus * cpu_usage / 100))
    nproc = int(avail_cpus * fraction)
    return max(nproc, 1)


def read_trajectory(trj_in):
    """
    Load a Desmond trajectory directory and return its frame generator.
    Each frame object has methods like .pos() and .time.
    """
    return traj.read_traj(trj_in)


def read_cms_file(cms_file):
    """
    Load and return (msys_model, cms_model) from a Desmond -out.cms file.
    """
    msys_model, cms_model = topo.read_cms(cms_file)
    return msys_model, cms_model


def parse_slice(slice_str, n_frames):
    """
    Given a slice specifier "start:end:step" and total frame count n_frames,
    return a Python range(start, end_idx, step), where
      • if end <= 0, interpret end as “go to the last frame” (n_frames).
      • otherwise end_idx = end.
    """
    start, end, step = map(int, slice_str.split(':'))
    if end <= 0:
        end_idx = n_frames
    else:
        end_idx = end
    return range(start, end_idx, step)


def get_atom_ids(cms_model, pairs):
    """
    Convert a list of atom‐index pairs (zero-based indices into cms_model.atom)
    into human-readable labels like "A_10_ALA_CA/A_12_GLY_CA".
    """
    atoms = [a for a in cms_model.atom]
    atom_ids = []
    for (i, j) in pairs:
        a1, a2 = atoms[i], atoms[j]
        ch1, ch2 = a1.chain.strip(), a2.chain.strip()
        rn1, rn2 = a1.resnum, a2.resnum
        rname1, rname2 = a1.pdbres.strip(), a2.pdbres.strip()
        aname1, aname2 = a1.pdbname.strip(), a2.pdbname.strip()
        atom_ids.append(
            f"{ch1}_{rn1}_{rname1}_{aname1}/"
            f"{ch2}_{rn2}_{rname2}_{aname2}"
        )
    return atom_ids


def get_atom_ids_solo(cms_model, gids):
    """
    Convert a list of GIDs into "chain resnum resname atomname" labels.
    """
    atoms = [a for a in cms_model.atom]
    atom_ids = []
    for g in gids:
        a = atoms[g]
        atom_ids.append(f"{a.chain.strip()} {a.resnum} {a.pdbres.strip()} {a.pdbname.strip()}")
    return atom_ids


def write_matrix_csv(out_path, cms_model, gids, matrix):
    """
    Write a CSV for a single (n_atoms x n_atoms) distance matrix,
    including row and column labels.

    Parameters
    ----------
    out_path : str
        The desired path (or prefix) for the CSV. If it doesn’t already
        end with ".csv", this function will append ".csv" automatically.
    cms_model : <schrodinger.cms.CmsModel>
        The CMS model (so we can look up chain/resnum/resname/atomname).
    gids : list[int]
        The list of GIDs (zero-based) for all selected atoms (length = n_atoms).
        We’ll use `get_atom_ids_solo(cms_model, gids)` to compute labels like
        "A 10 ALA CA".
    matrix : numpy.ndarray or cupy.ndarray
        A square array of shape (n_atoms, n_atoms) containing distances.
        If this is a CuPy array, the function will convert to NumPy internally.

    Output
    ------
    Writes a CSV whose first row is:
        ["", atom_label_1, atom_label_2, ..., atom_label_n]
    and each subsequent row i is:
        [atom_label_i, matrix[i, 0], matrix[i, 1], ..., matrix[i, n-1]]
    """
    import os
    import csv

    # Ensure it ends in .csv
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".csv":
        out_path = base + ".csv"

    # Convert CuPy→NumPy if needed
    if isinstance(matrix, cp.ndarray):
        matrix = cp.asnumpy(matrix)

    n_atoms = matrix.shape[0]
    if matrix.shape[1] != n_atoms:
        raise ValueError(f"Matrix must be square; got shape {matrix.shape}.")

    # Build single-atom labels via get_atom_ids_solo
    atom_labels = get_atom_ids_solo(cms_model, gids)

    # Write CSV
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)

        # First row: blank corner + all column labels
        writer.writerow([""] + atom_labels)

        # Each subsequent row: (row_label, row_values...)
        for i, row_label in enumerate(atom_labels):
            row_vals = matrix[i, :].tolist()
            writer.writerow([row_label] + row_vals)

    print(f"[CSV] Wrote square matrix ({n_atoms}×{n_atoms}) → {out_path}")


def write_distances_csv(out_path, atom_pair_labels, all_data, n_reps):
    """
    Write a CSV at out_path (force .csv extension) with columns:
      Pair, Rep1, Rep2, …, RepN, Average

    - atom_pair_labels: list of strings, length = (n_atoms * n_atoms), for ASL-selected atoms.
    - all_data:         2D NumPy array of shape ((n_atoms*n_atoms), n_reps+1),
                        where columns 0..(n_reps-1) are per-replicate distances,
                        column n_reps is the average across replicates.
    - n_reps:           number of replicates
    """
    base, ext = os.path.splitext(out_path)
    if ext.lower() != '.csv':
        out_path = base + '.csv'

    headers = ['Pair'] + [f"Rep{r+1}" for r in range(n_reps)] + ['Average']
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)
        for idx, pair_label in enumerate(atom_pair_labels):
            row = [pair_label] + all_data[idx].tolist()
            writer.writerow(row)

    print(f"[CSV] Wrote distances CSV ({len(atom_pair_labels)} rows × {n_reps+2} cols) to {out_path}")


def write_average_only_csv(out_path, atom_pair_labels, all_data):
    """
    Write out a CSV with exactly two columns: Pair, AverageDistance.

    Parameters
    ----------
    out_path : str
        Path (or prefix) for the CSV. If it doesn’t already end in ".csv",
        this function will append ".csv" automatically.
    atom_pair_labels : list[str]
        Length = n_atoms*n_atoms. Each entry is a string like
        "A_10_ALA_CA/A_12_GLY_CA", specifically for ASL-selected atoms.
    all_data : numpy.ndarray or cupy.ndarray
        A 2D array of shape (n_atoms*n_atoms, n_reps+1).
        The last column (index = n_reps) is assumed to be the “Average”
        distance across replicates.
    """
    import os
    import csv

    # Ensure .csv extension
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".csv":
        out_path = base + ".csv"

    # Convert CuPy→NumPy if needed
    if isinstance(all_data, cp.ndarray):
        all_data = cp.asnumpy(all_data)

    n_rows = all_data.shape[0]
    avg_col = all_data[:, -1]

    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Pair", "Average"])
        for i in range(n_rows):
            writer.writerow([atom_pair_labels[i], float(avg_col[i])])

    print(f"[CSV] Wrote average‐only CSV ({n_rows} rows) → {out_path}")


def get_parser():
    """
    Build and return a Namespace of parsed arguments for this script.
    """
    desc = (
        "This script calculates the intraprotein Cα distances (N×N matrix), "
        "using only atoms selected by ASL. It accepts one or more "
        "Desmond trajectory directories. It always writes out three CSVs: "
        "1) average square matrix, 2) per‐replicate + average, 3) average only."
    )
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Desmond trajectory directories (one or more).'
    )
    parser.add_argument(
        '-cms_file',
        required=True,
        type=str,
        help='Path to Desmond -out.cms file'
    )
    parser.add_argument(
        '-outname',
        required=True,
        type=str,
        help='Prefix for the output CSVs (e.g. distances)'
    )
    parser.add_argument(
        '-N',
        type=float,
        default=0.5,
        help=(
            "CPU cores to use. If >=1, uses exactly that many. "
            "If <1, uses that fraction of currently available CPUs."
        )
    )
    parser.add_argument(
        '-asl',
        type=str,
        default='protein and a. CA',
        help='Maestro ASL for selecting Cα atoms (default: protein and a. CA)'
    )
    parser.add_argument(
        '-s',
        type=str,
        default="0:-1:1",
        help=(
            "Frame slice specifier as start:end:step (e.g. 0:100:2). "
            "Default = 0:-1:1 (every frame)."
        )
    )
    return parser.parse_args()


def main():
    args = get_parser()
    start_time = datetime.now()

    # ────────── Load CMS & convert ASL to GIDs ──────────
    msys_model, cms_model = read_cms_file(args.cms_file)
    gids = topo.asl2gids(cms_model, args.asl)

    n_atoms = len(gids)
    if n_atoms == 0:
        sys.exit(f"ERROR: ASL '{args.asl}' returned no atoms; please check your ASL string or CMS file.")

    n_reps = len(args.infiles)
    per_rep_results = np.zeros((n_atoms, n_atoms, n_reps), dtype=np.float32)

    # ────────── Process each trajectory replicate ──────────
    for rep_idx, trj_dir in enumerate(args.infiles, start=1):
        trj_gen = read_trajectory(trj_dir)

        # Collect the positions of ONLY those GIDs for every frame
        frame_positions = []  # each entry: (n_atoms, 3)
        for frame_idx, frame in enumerate(trj_gen):
            pos_sel = frame.pos(gids)    # <--- IMPORTANT: pos(gids)
            frame_positions.append(pos_sel)

        n_frames = len(frame_positions)
        print(f"[info] Replicate {rep_idx}/{n_reps}: loaded {n_frames} frames from {trj_dir}")

        if n_frames == 0:
            sys.exit(f"ERROR: No frames found in '{trj_dir}'")

        # Apply slicing
        sl = parse_slice(args.s, n_frames)
        coords_sel = [frame_positions[i] for i in sl]
        n_sel_frames = len(coords_sel)
        print(f"[info] After slicing '{args.s}', using {n_sel_frames} frames.")

        if n_sel_frames == 0:
            sys.exit(f"ERROR: After slicing, no frames remain in '{trj_dir}'")

        # Determine batch size and number of batches
        batch = 10 if n_sel_frames >= 10 else n_sel_frames
        total_batches = (n_sel_frames + batch - 1) // batch

        # Accumulator on GPU
        sum_d_gpu = cp.zeros((n_atoms, n_atoms), dtype=cp.float32)

        # ─── GPU‐batched distance summation with progress ───
        for batch_idx, start in enumerate(range(0, n_sel_frames, batch), start=1):
            end = min(start + batch, n_sel_frames)
            print(f"[GPU] Rep {rep_idx}: batch {batch_idx}/{total_batches} "
                  f"(frames {start+1}–{end} of {n_sel_frames})")

            # Stack onto GPU; each coords_sel[k] is already (n_atoms,3)
            gpu_batch = cp.array(np.stack(coords_sel[start:end]), dtype=cp.float32)
            # gpu_batch shape = (batch_size, n_atoms, 3)

            # Compute pairwise differences
            diff = gpu_batch[:, :, None, :] - gpu_batch[:, None, :, :]  # (b, n_atoms, n_atoms, 3)
            dists = cp.linalg.norm(diff, axis=-1)                       # (b, n_atoms, n_atoms)

            # Sum over this batch
            sum_d_gpu += cp.sum(dists, axis=0)

            # Cleanup
            del gpu_batch, diff, dists
            cp._default_memory_pool.free_all_blocks()

        # Average for this replicate
        avg_d_gpu = sum_d_gpu / float(n_sel_frames)
        per_rep_results[:, :, rep_idx-1] = cp.asnumpy(avg_d_gpu).astype(np.float32)
        print(f"[info] Replicate {rep_idx} done (averaged over {n_sel_frames} frames).\n")

    # ───────── After all replicates, build data for CSVs ─────────
    # 1) Square average matrix (n_atoms x n_atoms)
    avg_across = np.mean(per_rep_results, axis=2)  # shape = (n_atoms, n_atoms)

    # 2) “Long” data for per‐replicate + average:
    flattened_reps = [per_rep_results[:, :, r].reshape(n_atoms * n_atoms) for r in range(n_reps)]
    flattened_avg = avg_across.reshape(n_atoms * n_atoms)
    all_data = np.column_stack(flattened_reps + [flattened_avg])

    # 3) Build atom‐pair labels of length n_atoms*n_atoms (using only ASL atoms)
    #    We pair gids[i] and gids[j] so that get_atom_ids uses the correct indices
    pairs = [(gids[i], gids[j]) for i in range(n_atoms) for j in range(n_atoms)]
    atom_pair_labels = get_atom_ids(cms_model, pairs)  # now only ASL atoms

    # ─────────── Write CSV #1: average square matrix ───────────
    avg_matrix_prefix = f"{args.outname}_Average_matrix"
    write_matrix_csv(avg_matrix_prefix, cms_model, gids, avg_across)

    # ─────────── Write CSV #2: per‐replicate + average ───────────
    perrep_prefix = f"{args.outname}_PerRep_vs_Average"
    write_distances_csv(perrep_prefix, atom_pair_labels, all_data, n_reps)

    # ─────────── Write CSV #3: average‐only CSV ───────────
    avg_only_prefix = f"{args.outname}_AverageOnly"
    write_average_only_csv(avg_only_prefix, atom_pair_labels, all_data)

    print("Total runtime:", datetime.now() - start_time)


if __name__ == '__main__':
    main()
