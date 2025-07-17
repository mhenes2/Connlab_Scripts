from schrodinger.application.desmond.packages import traj, topo, analysis
import csv
from datetime import datetime
import os
from collections import Counter
from itertools import chain
import argparse  # For parsing command-line options


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

    protein_ligand_interactions_df = protein_protein_interactions(msys_model, cms_model, flat_frames, args.combined_asl)

    ligand_ligand_interactions_df = protein_protein_interactions(msys_model, cms_model, flat_frames, args.ligand_asl)



# hbond_bb = []
# hbond_ss = []
# hbond_sb = []
# hbond_bs = []
#
# for frame in range(len(results)):
#     for key in results[frame]:
#         globals()[key].append(results[frame][key])
#
#
# hbond_bb = list(itertools.chain(*hbond_bb))
# hbond_ss = list(itertools.chain(*hbond_ss))
# hbond_sb = list(itertools.chain(*hbond_sb))
# hbond_bs = list(itertools.chain(*hbond_bs))
#
# hbond_bb_gid = [(x + 1, y + 1) for (x, y) in hbond_bb]
# hbond_ss_gid = [(x + 1, y + 1) for (x, y) in hbond_ss]
# hbond_sb_gid = [(x + 1, y + 1) for (x, y) in hbond_sb]
# hbond_bs_gid = [(x + 1, y + 1) for (x, y) in hbond_bs]
#
# hbond_bb_counter = Counter(hbond_bb_gid)
# hbond_ss_counter = Counter(hbond_ss_gid)
# hbond_sb_counter = Counter(hbond_sb_gid)
# hbond_bs_counter = Counter(hbond_bs_gid)
#
# hbond_bb_filtered = []
# hbond_ss_filtered = []
# hbond_sb_filtered = []
# hbond_bs_filtered = []
#
# for item, count in hbond_bb_counter.items():
#     if (count / float(len(trj_out))) * 100. > 50.:
#         # split it up
#         info = get_atom_ids_solo(cms_model,item)
#         hbond_bb_filtered.append([info, float(count)])
#
# with open('test_bb_hbond.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(hbond_bb_filtered)
#
#
# for item, count in hbond_ss_counter.items():
#     if (count / float(len(trj_out))) * 100. > 50.:
#         # split it up
#         info = get_atom_ids_solo(cms_model,item)
#         hbond_ss_filtered.append([info, float(count)])
#
# with open('test_ss_hbond.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(hbond_ss_filtered)
#
#
# for item, count in hbond_sb_counter.items():
#     if (count / float(len(trj_out))) * 100. > 50.:
#         # split it up
#         info = get_atom_ids_solo(cms_model,item)
#         hbond_sb_filtered.append([info, float(count)])
#
# with open('test_sb_hbond.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(hbond_sb_filtered)
#
#
# for item, count in hbond_bs_counter.items():
#     if (count / float(len(trj_out))) * 100. > 50.:
#         # split it up
#         info = get_atom_ids_solo(cms_model,item)
#         hbond_bs_filtered.append([info, float(count)])
#
# with open('test_bs_hbond.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(hbond_bs_filtered)

if __name__ == '__main__':
    main()
