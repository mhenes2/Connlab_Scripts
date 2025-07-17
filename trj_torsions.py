import pandas as pd
import numpy as np
from schrodinger.application.desmond.packages import topo, traj, analysis


from datetime import datetime
import argparse


def parse_args():
    """
    Argument parser when script is run from commandline
    :return:
    """
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    parser.add_argument('-cms', dest='cms_file', required=True,
                        help='Path to the Desmond -out.cms file')
    parser.add_argument('-o', dest='outname', required=True,
                        help='Base name for output CSV files')
    parser.add_argument('-torsions', dest='torsions_list', required=False,
                        help='list of torsion angles to compute. Supported names are: '
                             'Phi, Psi, Omega, Chi1, Chi2, Chi3, Chi4, Chi5'
                             'Default is Phi and Psi', type=list,
                        default=["Phi", "Psi"])
    parser.add_argument('-e', dest='entropy', required=False,
                        help='Entropy of torsion angles to compute',
                        default=True)
    parser.add_argument('-c', dest='chains', required=False,
                        help='Protein ASL to compute torsion angles', type=list,
                        default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
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


def shannon_entropy(angles, num_bins=36, base=2):
    """
    Compute the Shannon entropy of torsion angles.

    Parameters:
    angles (array-like): List or array of angles in degrees.
    num_bins (int): Number of bins to use for histogram. Default is 36 (10° bins).
    base (int or str): Logarithm base. Use 2 for bits or 'e' for nats. Default bits.

    Returns:
    float: Shannon entropy.
    """
    # Compute histogram of angles over the range -180 to 180 degrees
    hist, _ = np.histogram(angles, bins=num_bins, range=(-180, 180))
    probs = hist / np.sum(hist)
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]

    # Select logarithm function based on base
    if base == 'e':
        log_fn = np.log
    else:
        log_fn = np.log2

    # Shannon entropy formula
    return -np.sum(probs * log_fn(probs))


def main():
    args = parse_args()
    start_time = datetime.now()

    msys_model, cms_model = read_models(args.cms_file)

    # empty list to keep the results of the len(args.infiles) in
    all_reps = []

    for idx, trj in enumerate(args.infiles, start=1):
        frames = read_traj(trj)
        per_rep_results = {}
        for chain in args.chains:
            # get the full list of residue identifiers
            residues = list(cms_model.chain[chain].residue)
            # skip the first and last residue of the chain as they don't contain Phi and Psi angle
            for res_id in residues[1:-1]:
                residue = cms_model.findResidue(str(res_id))
                for torsion in args.torsions_list:
                    dihed_atoms = residue.getDihedralAtoms(torsion)
                    readable_dihed_atoms = [atom.index for atom in dihed_atoms]
                    torsion_analysis = analysis.Torsion(msys_model, cms_model, *readable_dihed_atoms)
                    torsion_analysis_results = analysis.analyze(frames[:5], torsion_analysis)
                    per_rep_results['{} - {}'.format(torsion, res_id)] = torsion_analysis_results

        per_rep_data = pd.DataFrame(list(per_rep_results.values()), index=list(per_rep_results.keys())).T

        # append the per rep data to the all reps list
        all_reps.append(per_rep_data)

        # write the per rep data to CSV
        per_rep_data.to_csv("{}_rep{}_torsions.csv".format(args.outname, idx), index_label='frame #')

    # stack the per rep data along axis 2
    results = np.stack(all_reps, axis=2)

    results_average = np.mean(results, axis=2)

    avg_df = pd.DataFrame(results_average, index=per_rep_data.index, columns=per_rep_data.columns)
    avg_df.to_csv(f"{args.outname}_average_torsions.csv", index_label='frame #')

    # Compute entropy column‐wise
    entropies = {}
    for col in avg_df.columns:
        angles = avg_df[col].dropna().values
        ent = shannon_entropy(
            angles,
            num_bins=36,
            base=2)
        entropies[col] = ent

    # Build result DataFrame
    ent_df = pd.DataFrame.from_dict(
        entropies, orient='index', columns=['entropy']
    )
    ent_df.index.name = 'torsion'

    # Output
    if args.outname:
        ent_df.to_csv("{}_entropy.csv".format(args.outname))
        print(f"Entropy values written to {args.outname}")
    else:
        print(ent_df)

    duration = datetime.now() - start_time
    print(f"Total runtime: {duration}")

if __name__ == '__main__':
    main()