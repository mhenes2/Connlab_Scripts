from schrodinger.application.desmond.packages import traj, topo, analysis
from schrodinger.application.desmond import cms
import itertools
import numpy as np
from schrodinger import structure
from datetime import datetime  # To measure runtime duration
import logging   # For logging progress and info
import argparse  # For parsing command-line options
import gc
import os
import shutil


# Configure logging to output timestamps and log levels
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="To strip a Desmond trj of water, membrane or ions"
    )
    # Input trajectory directories (one or more)
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    # Required path to CMS model
    parser.add_argument('-cms', required=True,
                        help='Path to the Desmond -out.cms file')
    parser.add_argument('--name', type=str, default='nowater',
                        help='Prefix for output files')
    parser.add_argument('--asl', type=str, default='not water and not r.ptype CL and not r.ptype NA and not membrane',
                        help='asl of atoms to extract')
    parser.add_argument('--chunks', type=int, default=5,
                        help='Number of chunks to break trajectory into')
    parser.add_argument('--concat', action='store_true', dest='do_concat',
                        help='Concatenate the stripped trj')
    return parser.parse_args()


def read_traj(trj_path) -> list:
    """
    Read trajectory frames from a Desmond directory.
    """
    # Convert generator to list for multiple passes
    return list(traj.read_traj(str(trj_path)))


def read_models(cms_path):
    """
    Load MSYS and CMS models for reference.
    """
    # Parse CMS file to get MSYS and CMS models
    msys_model, cms_model = topo.read_cms(cms_path)
    return msys_model, cms_model


def new_cms(asl, output_name, cms_model, index):
    stripped_system = topo.extract_subsystem(cms_model, asl)
    cms.Cms.write(stripped_system[0], '{}_stripped_rep{}-out.cms'.format(output_name, index))


def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def extract_system(output_name, trj_in, chunks, index):
    sub_msys_model, sub_cms_model = topo.read_cms('{}_stripped_rep{}-out.cms'.format(output_name, index))

    trj_out = traj.read_traj(trj_in)

    trj = list(split(trj_out, 4002))

    intermediates = []
    for i in range(chunks):
        extract = topo.asl2gids(sub_cms_model, 'all', include_pseudoatoms=True)
        traj.write_traj(traj.extract_subsystem(trj[i], extract), fname='{}_rep{}_{}_trj'
                        .format(output_name, index, i))
        intermediates.append(str('{}_rep{}_{}_trj'.format(output_name, index, i)))

    return intermediates

def concatenate_lists(list_of_lists):
    concatenated_list = []
    for sublist in list_of_lists:
        concatenated_list += sublist
    return concatenated_list


def concatenate_trj(chunks, output_name, index):
    x = []
    for i in range(chunks):
        trj_in = '{}_rep{}_{}_trj/'.format(output_name, index, i)
        trj_out = traj.read_traj(trj_in)
        x.append(trj_out)

    traj.write_traj(traj.merge(concatenate_lists(x)), fname='{}_stripped_rep{}_trj'.format(output_name, index))


def del_tmpfiles(dirs):
    for d in dirs:
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                print(f"Deleted directory {d!r}")
            except Exception as e:
                print(f"Error deleting {d!r}: {e}")
        else:
            print(f"Not a directory or doesn’t exist: {d!r}")


def main():
    """
    Main routine: orchestrates parsing, analysis, and output.
    """
    args = parse_args()         # Read command-line inputs
    start_time = datetime.now() # Start timer

    # Load models and structure for ASL
    msys_model, cms_model = read_models(args.cms)

    for idx, trj_dir in enumerate(args.infiles, start=1):
        tmpfiles = []
        # write out the new cms file first
        new_cms(args.asl, args.name, cms_model, idx)

        tmpfiles.append(extract_system(args.name, trj_dir, args.chunks, idx))

        concatenate_trj(args.chunks, args.name, idx)

    # if args.do_concat:
    #     trajs = []
    #     for f in args.infiles:
    #         tr = traj.read_traj(f)
    #         trajs.append(tr)
    #     result = list(traj.concat(*trajs))
    #     print('The resultant trajectory has {} frames.'.format(len(result)))
    #     traj.write_traj(result, fname='{}_stripped_concat_trj'.format(args.name))
    #     out_cms_fname = args.output_name + 'concat-out.cms'
    #     cms_model.fix_filenames(out_cms_fname, '{}_stripped_concat_trj'.format(args.name))
    #     cms_model.write(out_cms_fname)

    # flatten a list of lists
    tmpfiles = [item for sublist in tmpfiles for item in sublist]

    #cleanup
    del_tmpfiles(tmpfiles)

    # Log total runtime
    duration = datetime.now() - start_time
    logger.info(f"Total runtime: {duration}")


if __name__ == '__main__':
    main()