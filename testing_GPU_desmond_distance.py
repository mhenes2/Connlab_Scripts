# import os
from schrodinger.application.desmond.packages import traj, topo
import itertools
import numpy as np
from scipy.spatial.distance import euclidean
from multiprocessing import Pool
import psutil
from functools import partial
from datetime import datetime
import csv
import argparse
import cupy as cp


# call("export SCHRODINGER_ALLOW_UNSAFE_MULTIPROCESSING=1")
# this is needed to allow for multiprocessing
# os.system("export SCHRODINGER_ALLOW_UNSAFE_MULTIPROCESSING=1")

# get the time the script starts running
startTime = datetime.now()

# ASL = 'protein and a. CA'

# From Flo
def dynamic_cpu_assignment(fraction):
    """
    Use a fraction of the available cpus
    :param fraction: <float> The fraction of the total cpus to use
    :return:
    """
    if fraction >= 1:
        return int(fraction)
    # get number of cpus
    ncpus = psutil.cpu_count()
    # get cpu usage off a 2s window
    cpu_usage = psutil.cpu_percent(interval=2)
    # available cpus is only an approximation of the available capacity
    avail_cpus = int(ncpus - (ncpus * cpu_usage / 100))
    nproc = int(avail_cpus * fraction)
    if nproc == 0:
        nproc = 1
    return nproc


# This function reads the trajectory and returns the trj_out (a frame list)
def read_trajectory(trj_in):
    # Load the trajectory
    trj_out = traj.read_traj(trj_in)
    # return the trajectory object
    return trj_out


# This function reads the cms file and returns the msys_model and cms_model
def read_cms_file(cms_file):
    # Load the cms_model
    msys_model, cms_model = topo.read_cms(cms_file)
    # return the msys_model and cms_model
    return msys_model, cms_model


def get_dist(frame_coord, pairs=None, func=None):
    distance_vec = np.zeros(len(pairs))

    if func is None:
        func = euclidean
    if pairs is None:
        return

    for n, (a, b) in enumerate(pairs):
        coord1 = frame_coord[a]
        coord2 = frame_coord[b]
        dist = func(coord1, coord2)
        distance_vec[n] = dist
    return distance_vec


def get_atom_ids(cms_model, pairs):
    # to get the info of the pairs
    atoms = [a for a in cms_model.atom]
    atom_ids = []

    counter = 0
    while counter < len(pairs):
        a1 = atoms[pairs[counter][0]]
        a2 = atoms[pairs[counter][1]]
        chain1 = a1.chain.strip()
        chain2 = a2.chain.strip()
        resnum1 = a1.resnum
        resnum2 = a2.resnum
        resname1 = a1.pdbres.strip()
        resname2 = a2.pdbres.strip()
        atom_name1 = a1.pdbname.strip()
        atom_name2 = a2.pdbname.strip()
        atom_ids.append('{}_{}_{}_{}/{}_{}_{}_{}'.format(chain1, resnum1, resname1, atom_name1,
                                                         chain2, resnum2, resname2, atom_name2))
        counter = counter + 1

    return atom_ids


def get_atom_ids_solo(cms_model, gids):
    # to get the info of the pairs
    atoms = [a for a in cms_model.atom]
    atom_ids = []

    counter = 0
    while counter < len(gids):
        a1 = atoms[gids[counter]]
        chain1 = a1.chain.strip()
        resnum1 = a1.resnum
        resname1 = a1.pdbres.strip()
        atom_name1 = a1.pdbname.strip()
        atom_ids.append('{} {} {} {}'.format(chain1, resnum1, resname1, atom_name1))
        counter = counter + 1

    return atom_ids


def get_parser():
    script_desc = "This script calculates the intraprotein c-alpha distances. Generates a matrix of N x N residues. " \
                  "This script accepts a concatenated trj or individual trjs. If given individual trjs, it will " \
                  "first perform the concatenation then perform the c-alpha distance calculation"

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infiles',
                        help='desmond trajectory directories to be used in RMSD and protein/ligand RMSF calculations',
                        nargs='+')
    parser.add_argument('-cms_file',
                        help='desmond -out.cms',
                        type=str,
                        required=True)
    parser.add_argument('-outname',
                        help='name of the job',
                        type=str,
                        required=True)
    parser.add_argument('-N',
                        help="Number of cores to use. If whole number, that number of cores will be used."
                             "If fraction, that fraction of cores from the number of available cores will be used."
                             "Default is to use 1/2 of available CPUs",
                        default=0.5,
                        type=int)
    parser.add_argument('-asl',
                        help="",
                        default='protein and a. CA',
                        type=str)
    parser.add_argument('-s',
                        help="Use every nth frame of the trajectory. Default is 1 (i.e., use every frame in the trj."
                             "Start:End:Step",
                        type=str,
                        default="0:-1:1")
    return parser


def main(args):
    # Call the read_cms_file function, pass to it the *-out.cms file (script input)
    # Assign the 2 outputs the names msys_model and cms_model
    msys_model, cms_model = read_cms_file(args.cms_file)

    # get the global ids (gid) for the ASL to measure distances for
    gid = topo.asl2gids(cms_model, args.asl)

    per_rep_results = np.zeros((len(gid), len(gid), len(args.infiles)))

    # Loop over every trj passed to the script
    for index, trajectory in enumerate(args.infiles):
        # read the input trj
        trj = read_trajectory(trajectory)
        print("Done loading trajectory {}".format(index + 1))

        # get the frame coordinates for euclidean (xyz) distance measurement
        frame_coords = [f.pos() for f in trj]
        n_frames, n_atoms = len(frame_coords), len(frame_coords[1])

        # decide batch size
        # batch = 100 or n_frames
        batch = 10

        # allocate accumulator on GPU
        sum_d = cp.zeros((n_atoms, n_atoms), dtype=cp.float16)

        # process in batches
        for start in range(0, n_frames, batch):
            end = min(start + batch, n_frames)
            # move this batch to GPU: shape (b, n_atoms, 3)
            coords_batch = cp.array(frame_coords[start:end], dtype=cp.float16)

            # compute pairwise diffs in one go:
            #   coords_batch[:, :, None, :] has shape (b, n_atoms, 1, 3)
            #   coords_batch[:, None, :, :] has shape (b, 1, n_atoms, 3)
            diff = coords_batch[:, :, None, :] - coords_batch[:, None, :, :]  # (b, n, n, 3)

            # L2 norm along the last axis → (b, n, n)
            dists = cp.linalg.norm(diff, axis=-1)

            # sum over frames in this batch → (n, n), then accumulate
            sum_d += cp.sum(dists, axis=0)

            # free GPU memory for this batch
            del coords_batch, diff, dists
            cp._default_memory_pool.free_all_blocks()

        # average over all frames
        avg_d = sum_d / float(n_frames)

        # bring back to CPU and save
        avg_d_np = cp.asnumpy(avg_d)
        np.savetxt(args.outname, avg_d_np, delimiter=",")
        print(f"[GPU-batched] Wrote average distance matrix ({n_atoms}×{n_atoms}) to {args.outname}")


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)
