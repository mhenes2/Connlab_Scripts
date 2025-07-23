from schrodinger.application.desmond.packages import traj, topo, analysis
import csv
from datetime import datetime
import os
from collections import Counter
from itertools import chain
import argparse  # For parsing command-line options
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return namespace.
    """
    parser = argparse.ArgumentParser(
        description=""
    )
    # Input trajectory directories (one or more)
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    # Required path to CMS model
    parser.add_argument('-cms', dest='cms_file', required=True,
                        help='Path to the Desmond -out.cms file')
    # Output base name for CSV files
    parser.add_argument('-o', dest='outname', required=True,
                        help='Base name for output CSV files')
    # Optional ASL for protein selection
    parser.add_argument('-protein_asl', '--protein_asl', dest='protein_asl',
                        default='((chain.name A) OR (chain.name B) OR (chain.name C))',
                        help='Protein atom selection (Maestro ASL)')
    parser.add_argument('-combined_asl', '--combined_asl', dest='combined_asl',
                        default='((chain.name A) OR (chain.name B) OR (chain.name C) OR (chain.name D) OR (chain.name E) OR (chain.name F) OR (chain.name G) OR (chain.name H) OR (chain.name I))',
                        help='Protein atom selection (Maestro ASL)')
    parser.add_argument('-ligand_asl', '--ligand_asl', dest='ligand_asl',
                        default='((chain.name D) OR (chain.name E) OR (chain.name F) OR (chain.name G) OR (chain.name H) OR (chain.name I))',
                        help='Protein atom selection (Maestro ASL)')
    return parser.parse_args()


def read_traj(trj_path):
    """
    Read trajectory frames from a Desmond directory.
    """
    # Convert generator to list for multiple passes
    return traj.read_traj(trj_path)


def read_models(cms_path):
    msys_model, cms_model = topo.read_cms(cms_path)
    return msys_model, cms_model


def protein_protein_interactions(msys_model, cms_model, trj, protein_asl):
    protein_protein_interactions = analysis.ProtProtInter(msys_model, cms_model, asl=str(protein_asl))
    protein_protein_interactions_results = analysis.analyze(trj, protein_protein_interactions)

    # returns a dictionary with the following keys
    # dict_keys(['pi-cat', 'pi-pi', 'salt-bridge', 'hbond_bb', 'hbond_ss', 'hbond_sb', 'hbond_bs'])

    return protein_protein_interactions_results


def get_atom_ids_solo(cms_model, pairs):
    # to get the info of the pairs
    atoms = [a for a in cms_model.atom]
    a1 = atoms[pairs[0]]
    a2 = atoms[pairs[1]]
    chain1 = a1.chain.strip()
    chain2 = a2.chain.strip()
    resnum1 = a1.resnum
    resnum2 = a2.resnum
    resname1 = a1.pdbres.strip()
    resname2 = a2.pdbres.strip()
    atom_name1 = a1.pdbname.strip()
    atom_name2 = a2.pdbname.strip()
    x = '{}_{}_{}_{} - {}_{}_{}_{}'.format(chain1, resnum1, resname1, atom_name1,
                                                       chain2, resnum2, resname2, atom_name2)
    return x


def dicts_to_csv_pandas(dict_of_dicts, interaction=None, outname=None, trj_length=None):
    """
    For each key in dict_of_dicts, writes a CSV named
    "{outname}_{interaction}_{main_key}.csv" with columns:
      - pairs: formatted "A - B"
      - frequency: value as a percentage of trj_length

    Parameters:
    - dict_of_dicts: dict of dicts with tuple keys and count values
    - interaction:  string to include in filename
    - outname:      base name for output files
    - trj_length:   total frames/timepoints for calculating percentages
    """
    if trj_length is None or trj_length == 0:
        raise ValueError("trj_length must be provided and non-zero")

    for main_key, inner_dict in dict_of_dicts.items():
        # Build DataFrame
        df = pd.DataFrame(
            inner_dict.items(),
            columns=['pairs', 'frequency']
        )

        # Format the 'pairs' column
        df['pairs'] = df['pairs'].apply(lambda tup: f"{tup[0]} - {tup[1]}")

        # Convert counts to percentage of trj_length
        df['frequency'] = df['frequency'] / trj_length * 100

        # Write out CSV (with two decimal places)
        filename = f"{outname}_{interaction}_{main_key}.csv"
        df.to_csv(filename, index=False, float_format="%.2f")


def main():
    args = parse_args()

    # read the msys_model and cms_model from the cms_file provided
    msys_model, cms_model = topo.read_cms(args.cms_file)

    frames = []
    for trj_dir in args.infiles:
        # Load the trajectory
        frames.append(traj.read_traj(trj_dir))

    flat_frames = list(chain.from_iterable(frames))

    protein_protein_interactions_df = protein_protein_interactions(msys_model, cms_model, flat_frames, args.protein_asl)
    dicts_to_csv_pandas(protein_protein_interactions_df, "protein-protein", args.outname, len(flat_frames))

    protein_ligand_interactions_df = protein_protein_interactions(msys_model, cms_model, flat_frames, args.combined_asl)
    dicts_to_csv_pandas(protein_ligand_interactions_df, "protein-ligand", args.outname, len(flat_frames))

    ligand_ligand_interactions_df = protein_protein_interactions(msys_model, cms_model, flat_frames, args.ligand_asl)
    dicts_to_csv_pandas(ligand_ligand_interactions_df, "ligand-ligand", args.outname, len(flat_frames))

if __name__ == '__main__':
    main()
