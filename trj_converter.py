import mdtraj as md
import argparse
from datetime import datetime
import os
from schrodinger.application.desmond.packages import traj, topo, analysis
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


def get_parser():
    script_desc = ""

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infiles',
                        help='desmond trajectory directories that will be converted',
                        nargs='+')
    parser.add_argument('-top_file',
                        help='Topology file in PDB form',
                        type=str,
                        required=True)
    parser.add_argument('-outname',
                        help='name of the job. this name will be used to name the output files',
                        type=str,
                        required=True)
    parser.add_argument('-c',
                        help='concat trjs. Use this flag after you have outputted stripped trjs',
                        type=bool,
                        default=False)
    parser.add_argument('-extract_asl',
                        help='',
                        type=str,
                        default='protein',
                        required=False)
    parser.add_argument('-f',
                        help='output format. options are "dtr", "dcd", or "both" ',
                        type=str,
                        default="both")
    parser.add_argument('-cms_file',
                        help='cms file to be outputted without water',
                        type=str,
                        required=True)
    return parser


def main(args):
    # load the topology from the provided PDB file
    topology = md.load(args.top_file).topology

    # call the new_cms function to write out a new *-out.cms file with a subsystem
    new_cms(args.cms_file, args.extract_asl, args.outname)
    print ("Done writing no water CMS file")

    if args.c != True:
        if args.f == "dtr":
            for index, trajectory in enumerate(args.infiles):
                traj = md.load_dtr(trajectory, top=topology, atom_indices=topology.select(args.extract_asl))
                print(traj)

                print ('writing trj in dtr format')

                traj.save_dtr("{}_rep{}.dtr".format(args.outname, index + 1))
                print('Done saving dtr trajectory')

                # rename the output file to have _trj to match Desmond output
                os.rename("{}_rep{}.dtr".format(args.outname, index + 1), '{}_rep{}_trj'.format(args.outname, index + 1))

        elif args.f == "dcd":
            for index, trajectory in enumerate(args.infiles):
                traj = md.load_dtr(trajectory, top=topology, atom_indices=topology.select(args.extract_asl))
                print(traj)

                print ('writing trj in dcd format')

                traj.save_dcd("{}_rep{}.dcd".format(args.outname, index + 1))
                print('Done saving dcd trajectory')

        elif args.f == "both":
            for index, trajectory in enumerate(args.infiles):
                traj = md.load_dtr(trajectory, top=topology, atom_indices=topology.select(args.extract_asl))
                print(traj)

                print ('writing trj in dcd and dtr format')

                traj.save_dcd("{}_rep{}.dcd".format(args.outname, index + 1))
                print('Done saving dcd trajectory')

                traj.save_dtr("{}_rep{}.dtr".format(args.outname, index + 1))
                print('Done saving dtr trajectory')

                # rename the output file to have _trj to match Desmond output
                os.rename("{}_rep{}.dtr".format(args.outname, index + 1), '{}_rep{}_trj'.format(args.outname, index + 1))


    # if the user wants a concat trjs
    if args.c == True:
        trj_list = []
        for i in range(len(args.infiles)):
            trj_list.append(md.load(i))

        concat_trj = md.join(trj_list)

        # if concat, then only write out a dcd file
        concat_trj.save_dcd("{}_concat_nowater".format(args.outname))

        # rename the output file to have _trj to match Desmond output
        os.rename('{}_concat_nowater.dtr'.format(args.outname), '{}_concat_nowater_trj'.format(args.outname))


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)
