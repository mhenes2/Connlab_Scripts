import matplotlib
matplotlib.use('Agg')
from schrodinger.application.desmond.packages import traj, topo, analysis
from datetime import datetime
from schrodinger.structure import Structure
from schrodinger.structutils.analyze import evaluate_asl
import numpy as np
import argparse
from scipy.linalg import eigh


# get the time the script starts running
startTime = datetime.now()


# This function reads the cms file and returns the msys_model and cms_model
def read_cms_file(cms_file):
    # Load the cms_model
    msys_model, cms_model = topo.read_cms(cms_file)
    # return the msys_model and cms_model
    return msys_model, cms_model


# This function reads the trajectory and returns the trj_out (a frame list)
def read_trajectory(trj_in):
    # Load the trajectory
    trj_out = traj.read_traj(trj_in)
    # return the trajectory object
    return trj_out


# Align the trj to the backbone atoms
def align(msys_model, cms_model, trj, asl, st):
    aids = evaluate_asl(st, asl)
    fit_aids = evaluate_asl(st, 'backbone and not (atom.ele H)')
    fit_gids = topo.asl2gids(cms_model, 'backbone and not (atom.ele H)', include_pseudoatoms=True)
    fit_ref_pos = trj[0].pos(fit_gids)
    align_ = analysis.PosAlign(msys_model, cms_model, aids, fit_aids=fit_aids, fit_ref_pos=fit_ref_pos)
    aligned = analysis.analyze(trj, align_)
    average_st = np.mean(aligned, axis=0)

    return average_st


# Using the mean structure coordinates and the specified cuttoff, calculate the Hessian matrix
def compute_hessian_chunk(coords, cutoff):
    num_atoms = coords.shape[0]
    cutoff2 = cutoff ** 2
    hessian = np.zeros((num_atoms * 3, num_atoms * 3), dtype=np.float64)
    kirchhoff = np.zeros((num_atoms, num_atoms), dtype=np.float64)

    for i in range(num_atoms):
        res_i3 = i * 3
        res_i33 = res_i3 + 3
        i_p1 = i + 1
        i2j_all = coords[i_p1:, :] - coords[i]
        for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
            if dist2 > cutoff2:
                continue
            i2j = i2j_all[j]
            j += i_p1
            # g = gamma(dist2, i, j)
            g = 1.0
            res_j3 = j * 3
            res_j33 = res_j3 + 3
            super_element = np.outer(i2j, i2j) * (-g / dist2)
            hessian[res_i3:res_i33, res_j3:res_j33] = super_element
            hessian[res_j3:res_j33, res_i3:res_i33] = super_element
            hessian[res_i3:res_i33, res_i3:res_i33] -= super_element
            hessian[res_j3:res_j33, res_j3:res_j33] -= super_element
            kirchhoff[i, j] = -g
            kirchhoff[j, i] = -g
            kirchhoff[i, i] += g
            kirchhoff[j, j] += g

    return hessian, kirchhoff


# Linearize the Hessian matrix to calculate the modes
def calculate_modes(hessian):
    eigenvalues, eigenvectors = eigh(hessian)
    return eigenvalues, eigenvectors


# Calculate the new coordinates of a specific mode by adding the initial coordinates to the displacement
def generate_new_coordinates(initial_coords, displacement, mode_index):
    # Assuming initial_coords and displacement are numpy arrays of shape (n_atoms, 3)
    new_coords = initial_coords + displacement[:, mode_index].reshape(np.shape(initial_coords))

    return new_coords


# Calculate the RMSD of the eigenvector
def calculate_rmsd(eigenvector):
    """Calculate the root-mean-square deviation of the eigenvector displacements."""
    return np.sqrt(np.mean(np.square(eigenvector)))


# Choose a scale factor so that the maximum displacement is around the target displacement
def choose_scale_factor(eigenvector, target_displacement):
    """
    Choose a scale factor so that the maximum displacement is around the target displacement.
    target_displacement: Desired maximum displacement in Å (angstroms).
    """
    rmsd = calculate_rmsd(eigenvector)
    scale_factor = target_displacement / rmsd
    return scale_factor


# A function to normalize the eigenvectors
def normalize_eigenvectors(eigenvectors):
    # Normalizing eigenvectors
    norm_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    return norm_eigenvectors


# Parser function to take inputs from the user
def get_parser():
    script_desc = "This script will calculate the normal mode analysis from a trajectory. Use a concat trj"

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infiles',
                        help='input trj file. use a concat trj')
    parser.add_argument('-cms_file',
                        help='desmond -out.cms',
                        type=str,
                        required=True)
    parser.add_argument('-outname',
                        help='name to use for the output files',
                        type=str,
                        required=True)
    parser.add_argument('-asl',
                        help='The protein atomset in maestro atom selection language. default is all c-alpha atoms',
                        default='protein and a. CA',
                        type=str)
    parser.add_argument('-cutoff',
                        help='Distance cutoff for the Hessian matrix calculation',
                        default=15.0,
                        type=float)
    parser.add_argument('-target',
                        help='Desired maximum displacement in Å - basically ignore all structures above this cutoff',
                        default=2.0,
                        type=float)
    return parser


def main(args):
    # Call the read_cms_file function, pass to it the cms_file (script input)
    # Assign the 2 outputs the names msys_model and cms_model
    msys_model, cms_model = read_cms_file(args.cms_file)

    # load the cms file as a structure element
    st = Structure.read(args.cms_file)
    # select the atom ids for the calpha atoms
    protein_aids = cms_model.select_atom(args.asl)
    # extract the calpha atoms into a structure
    protein_st = cms_model.extract(protein_aids)

    # use the read_trajectory function to read the input trj - best to use a concat trj
    tr = read_trajectory(args.infiles)

    # Step 1: Compute the mean structure from an aligned trj
    mean_structure = align(msys_model, cms_model, tr, args.asl, st)

    # Step 2: Write the mean structure to a PDB file
    protein_st.setXYZ(mean_structure)
    protein_st.write('{}_mean_structure.pdb'.format(args.outname))

    # Step 3: Calculate hessian and kirchhoff matrices
    hessian, kirchhoff = compute_hessian_chunk(mean_structure, cutoff=args.cutoff)

    # Step 4: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_modes(hessian)

    # Step 5: normalize the eigenvectors
    norm_eigenvectors = normalize_eigenvectors(eigenvectors)

    # Step 6: Plot the first 20 modes to a structure
    for mode in range(20):

        # Step 6a: Select the mode to analyze and reshape into (n_atoms, 3)
        selected_mode_vector = norm_eigenvectors[:, mode].reshape((-1, 3))

        # Step 6b: Using the target displacement specified by the user,
        # find an appropriate scale vector for visualization
        scale_factor = choose_scale_factor(selected_mode_vector, args.target)

        # Step 6c: Multiply the selected mode by the appropriate scale factor
        selected_mode_vector *= scale_factor

        # Step 6d: Add the scaled mode vectors by the mean structure XYZ coordinates
        new_coordinates = mean_structure + selected_mode_vector

        # Step 6e: Plot the new coordinates to a structure and output to PDB
        protein_st.setXYZ(new_coordinates)
        protein_st.write('{}_Mode-{}.pdb'.format(args.outname,mode))


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)