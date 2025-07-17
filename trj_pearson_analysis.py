#!/usr/bin/env python3
"""
Parse output from `trj_nonbonded_transformer` and create
    • Square CSV matrix of averaged non‑bonded energies
    • Flattened single‑column CSV of the same data
    • Protein–protein and protein–ligand heat‑map figures

Usage (single trajectory):
    python nonbonded_parser.py run1_nonbonded.json -cms_file system-out.cms

Usage (average across replicates):
    python nonbonded_parser.py rep1.json rep2.json rep3.json -cms_file system-out.cms \
        -jobname my_run -protein_chains A B C
"""

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import schrodinger.application.desmond.packages.topo as topo

# -----------------------------------------------------------------------------
# Matplotlib setup (head‑less)
# -----------------------------------------------------------------------------
matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 18,
    "xtick.major.pad": 10,
    "ytick.major.pad": 4,
})

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_data_filtered(data_file, raw_data=None, chain=None, row_labels_dict=None):
    """Return/accumulate an *n × n* NumPy matrix, optionally filtered by *chain*."""

    # 1) Normalise chain selector to a *set* for fast lookup
    if chain is None:
        chains_to_keep = None
    else:
        chains_to_keep = {chain} if isinstance(chain, str) else set(chain)

    # 2) Read JSON produced by trj_nonbonded_transformer
    with open(data_file) as fh:
        data_dict = json.load(fh)

    # Atom‑IDs may be list or one long space‑delimited string
    atom_ids = (
        data_dict["atom_ids"].split()
        if isinstance(data_dict["atom_ids"], str)
        else list(data_dict["atom_ids"])
    )
    results = data_dict["results"]

    if len(atom_ids) != len(results):
        raise ValueError("Length mismatch between atom_ids and results list")

    # 3) Decide which rows/columns to keep
    if chains_to_keep is None:
        keep_idx = range(len(atom_ids))
    else:
        keep_idx = [i for i, aid in enumerate(atom_ids) if aid.split()[0] in chains_to_keep]
        if not keep_idx:
            raise RuntimeError("No atom IDs matched the requested chain(s)")

    # 4) Build square filtered matrix
    filt_ids = [atom_ids[i] for i in keep_idx]
    filt_mat = np.asarray([[results[i][j] for j in keep_idx] for i in keep_idx], dtype=float)

    # 5) First file → new matrix; later files → accumulate
    if row_labels_dict is None:
        row_labels_dict = {aid: idx for idx, aid in enumerate(filt_ids)}
        raw_data = filt_mat.copy()
        return raw_data, row_labels_dict

    # Consistency check
    if set(filt_ids) != set(row_labels_dict):
        diff = set(filt_ids) ^ set(row_labels_dict)
        raise RuntimeError(f"Atom IDs differ between files: {diff}")

    reorder = [row_labels_dict[aid] for aid in filt_ids]
    raw_data += filt_mat[np.ix_(reorder, reorder)]
    return raw_data, row_labels_dict


def plot_correlated_motions(
    data,
    outname,
    title="",
    xlabel="",
    ylabel="",
    column_labels=None,
    row_labels=None,
    hide_diagonal=True,
    color_scheme="jet",
    equal=False,
):
    """Generic heat‑map plotting wrapper."""

    cmap = plt.get_cmap(color_scheme)

    if hide_diagonal:
        diag_mask = np.eye(len(data), dtype=bool)
        data = np.ma.array(data, mask=diag_mask)
        cmap.set_bad(color="w", alpha=1)

    fig, ax = plt.subplots()
    heat = ax.pcolormesh(data, cmap=cmap, vmin=data.min(), vmax=data.max())

    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])

    fig.colorbar(heat)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, frame_on=False)

    if equal:
        ax.set_aspect("equal", adjustable="box")

    if column_labels is not None:
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=45, ha="right")
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, rotation=45)

    # ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(outname, dpi=300)
    plt.close(fig)


def write_matrix_csv(matrix: np.ndarray, labels: list[str], outname: str) -> None:
    """Write a square matrix with row/column labels to *outname*."""

    with open(outname, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def write_long_csv(matrix: np.ndarray, labels: list[str], outname: str) -> None:
    """Write a long-format CSV: *pair* (row-col) and value."""

    with open(outname, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "value"])
        for i, row_lab in enumerate(labels):
            for j, col_lab in enumerate(labels):
                writer.writerow([f"{row_lab} - {col_lab}", matrix[i, j]])


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Average non‑bonded interaction matrices, output CSVs, and make heat‑maps",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "infile",
        nargs="+",
        help="One or more *.json files from trj_nonbonded_transformer",
    )
    p.add_argument("-cms_file", required=True, help="Desmond system‑out.cms file")
    p.add_argument("-jobname", default="nonbonded_energy", help="Prefix for outputs")
    p.add_argument(
        "-protein_chains",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        help="One or more chain IDs making up the protein",
    )
    p.add_argument(
        "-ligand_asl",
        default="ligand and not a.element H",
        help="Maestro ASL that selects the ligand atoms",
    )

    return p


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(args):
    files_total = len(args.infile)

    raw_data, row_labels = None, None
    for f in args.infile:
        raw_data, row_labels = get_data_filtered(
            data_file=f,
            raw_data=raw_data,
            chain=args.protein_chains,
            row_labels_dict=row_labels,
        )

    raw_data /= files_total  # Average across replicates

    # Ordered labels
    ordered_labels = [lab for lab, _ in sorted(row_labels.items(), key=lambda kv: kv[1])]

    # CSV outputs: matrix and flattened vector
    write_matrix_csv(raw_data, ordered_labels, f"{args.jobname}.csv")
    write_long_csv(raw_data, ordered_labels, f"{args.jobname}_long.csv")

    # ------------------------------------------------------------------
    # Identify ligand atoms
    # ------------------------------------------------------------------
    msys_model, cms_model = topo.read_cms(args.cms_file)
    ligand_gids = topo.asl2gids(cms_model, args.ligand_asl)

    ligand_atoms = {
        f"{atm.chain.strip()} {atm.resnum} {atm.pdbres.strip()} {atm.pdbname.strip()}"
        for i, atm in enumerate(cms_model.atom)
        if i in ligand_gids
    }

    ligand_in_matrix = [aid for aid in ordered_labels if aid in ligand_atoms]
    if not ligand_in_matrix:
        raise RuntimeError(
            "No ligand atoms found in the interaction matrix – check ASL and chain filters"
        )

    protein_only = [aid for aid in ordered_labels if aid not in ligand_atoms]

    idx_map = row_labels  # atom‑ID → index
    prot_idx = [idx_map[aid] for aid in protein_only]
    lig_idx = [idx_map[aid] for aid in ligand_in_matrix]

    protein_mat = raw_data[np.ix_(prot_idx, prot_idx)]
    ligand_mat = raw_data[np.ix_(lig_idx, prot_idx)]

    # ------------------------------------------------------------------
    # Heat‑map plots
    # ------------------------------------------------------------------
    plot_correlated_motions(
        ligand_mat,
        f"{args.jobname}_ligand.png",
        title="Protein–ligand correlated motions",
        xlabel="Protein (C‑alpha)",
        ylabel="Ligand atoms",
        hide_diagonal=False,
    )

    plot_correlated_motions(
        protein_mat,
        f"{args.jobname}_protein.png",
        title="Protein–protein correlated motions",
        xlabel="Protein (C‑alpha)",
        ylabel="Protein (C‑alpha)",
    )


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main(get_parser().parse_args())
