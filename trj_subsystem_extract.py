import argparse
from datetime import datetime
from schrodinger.application.desmond.packages import traj, topo, analysis
from schrodinger import structure
from schrodinger.application.desmond import cms

startTime = datetime.now()


def new_cms(cms_file, asl, name):
    """
    This function will write out a new CMS file without waters
    :param str cms_file: file path for the input cms file
    :param str asl: atom language selection for the substructure you'd like to extract
    :param str name: output name. "_nowater-out.cms" will be added to the end of that name
    """

    msys_model, cms_model = topo.read_cms(cms_file)
    nowater = topo.extract_subsystem(cms_model, asl)
    cms.Cms.write(nowater[0], '{}_nowater-out.cms'.format(name))

    return msys_model, cms_model


def read_trajectory(trj_in):
    """
    This function will write out a new CMS file without waters
    :param str trj_in: file path for the input trj file
    :returns frame list
    """

    # Load the trajectory
    trj_out = traj.read_traj(trj_in)
    # return the trajectory object
    return trj_out


def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def concatenate_lists(list_of_lists):
    concatenated_list = []
    for sublist in list_of_lists:
        concatenated_list += sublist
    return concatenated_list


def get_parser():
    script_desc = ""

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infiles',
                        help='desmond trajectory directories that will be converted',
                        nargs='+')
    parser.add_argument('-cms_file',
                        help='cms file to be outputted without water',
                        type=str,
                        required=True)
    parser.add_argument('-outname',
                        help='name of the job. this name will be used to name the output files',
                        type=str,
                        required=True)
    parser.add_argument('-extract_asl',
                        help='',
                        type=str,
                        default='protein',
                        required=False)
    return parser


def main(args):
    msys_model, cms_model = new_cms(args.cms_file, args.extract_asl, args.outname)

    extract = topo.asl2gids(cms_model, 'protein', include_pseudoatoms=True)

    tr = read_trajectory(args.infiles)

    new_trj = []

    for i in list(split(tr, 1000)):
        new = []
        new = traj.extract_subsystem(i, extract)
        new_trj.append(new)

    final_trj = concatenate_lists(new_trj)

    traj.write_traj(final_trj, fname='{}_trj'.format(args.outname))

if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)
