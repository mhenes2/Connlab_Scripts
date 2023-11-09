from schrodinger.application.desmond.packages import traj, topo, analysis
import csv
from datetime import datetime
from schrodinger import structure
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.application.desmond import automatic_analysis_generator as auto
import numpy as np
import argparse

# get the time the script starts running - This is used so we can know how long things take to run
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

# Calculates the c-alpha RMSD and RMSF
# Needs the cms_model, msys_model, and the trajectory. Returns the results as an array
def calc_rmsd(optional_protein_asl, cms_model, msys_model, trj_out, st):
    # This is the ASL to use to calculate the RMSD
    # We default to c alpha atoms but the user can select another set of atoms
    if optional_protein_asl:
        ASL_calc = optional_protein_asl
    # if the user didn't provide a protein ASL, then default to the calpha
    else:
        ASL_calc = 'protein and a. CA'

    # if the system has a ligand, this function should get the ligand automatically
    ligand = auto.getLigand(st)
    # Get atom ids (aids) for the RMSD and RMSF calculation eg. if you want the RSMD for c-alpha carbons (CA), then
    # use 'a. CA' for the ASL_calc variable above
    # Documentation:
    # http://content.schrodinger.com/Docs/r2018-2/python_api/product_specific.html
    aids = cms_model.select_atom(ASL_calc)

    # Get the aids for the reference to be used in the RMSD calculation -
    # we don't need a ref aids just the position of the reference
    ref_aids = evaluate_asl(st, ASL_calc)
    # convert from aids to gids (global ids)
    ref_gids = topo.aids2gids(cms_model, ref_aids, include_pseudoatoms=False)
    # get the position for the reference gids from frame 1 - this is what we need
    ref_pos = trj_out[0].pos(ref_gids)
    print (np.shape(ref_pos))

    # get the fit aids position - we're going to find on the protein backbone
    if ligand[0] == None:
        # fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H) and not (({})))'.format(ligand[1]))
        fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H))')
    else:
        # fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H))')
        fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H) and not (({})))'.format(ligand[1]))
    # get the fit_aids gids
    fit_gids = topo.aids2gids(cms_model, fit_aids, include_pseudoatoms=False)
    # get the position of frame 1
    fit_ref_pos = trj_out[0].pos(fit_gids)
    print (np.shape(fit_ref_pos))

    # calculate RMSD
    # aids =
    # ref_pos =
    # fit_aids =
    # fit_ref_pos =
    rmsd_analysis = analysis.RMSD(msys_model=msys_model, cms_model=cms_model, aids=aids, ref_pos=ref_pos,
                                  fit_aids=fit_aids, fit_ref_pos=fit_ref_pos)

    results = (analysis.analyze(trj_out, rmsd_analysis, progress_feedback=analysis.progress_report_frame_number))
    return results


def get_parser():
    script_desc = "This script will calculate the protein RMSD/RMSF and ligand RMSF, if a ligand is in the structure, " \
                  "for any number of trajectories. The script allows for optional protein and ligand asl (see below)"

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infiles',
                        help= 'desmond trajectory directories to be used in RMSD and protein/ligand RMSF calculations',
                        nargs='+')
    parser.add_argument('-cms_file',
                        help='desmond -out.cms',
                        type=str,
                        required=True)
    parser.add_argument('-outname',
                        help='name to use for the output files',
                        type=str,
                        required=True)
    parser.add_argument('-protein_asl',
                        help='The protein atomset in maestro atom selection language. default is all c-alpha atoms',
                        default= 'protein and a. CA',
                        type=str)
    parser.add_argument('-ligand_asl',
                        help="The ligand atomset in maestro atom selection language. "
                             "By default, the script tries to find a ligand."
                             "If you would like to completely ignore the ligand, use 'APO' and an input",
                        type=str,
                        default='')
    # parser.add_argument('-s',
    #                     help="Use every nth frame of the trajectory. Default is 1 (i.e., use every frame in the trj."
    #                          "Start:End:Step",
    #                     type=str,
    #                     default="0:-1:1")
    # parser.add_argument('-a',
    #                     help="Analysis to compelete. Options. 'RMSD', 'RMSF','lig_RMSF'. Default is to only "
    #                          "calculate the RMSD ",
    #                     type=list,
    #                     default=["RMSD"])
    return parser


def main(args):
    # Call the read_cms_file function, pass to it the *-out.cms file (script input)
    # Assign the 2 outputs the names msys_model and cms_model
    msys_model, cms_model = read_cms_file(args.cms_file)

    st = structure.Structure.read(args.cms_file)

    # Load the first trj so you can get the length of the trajectory
    # don't need to load anything else bc all input trjs should be the same length
    tr = read_trajectory(args.infiles[0])
    # if args.s:
    #     tr = list(tr[i] for i in range(0, len(tr), args.s))

    # initialize the output arrays
    RMSD_output = np.zeros((len(tr), len(args.infiles)), dtype=np.float16)

    # delete to manage memory
    del tr

    traj_len = 0

    # iterate over the input trajectories
    for index, trajectory in enumerate(args.infiles):
        tr = read_trajectory(trajectory)
        print(len(tr))

        # # if the user wants to take strides, this will do that for them
        # if args.s:
        #     tr = list(tr[i] for i in range(0, len(tr), args.s))

        traj_len = len(tr)
        fr = tr[-1]
        print("Done loading trajectory {}".format(index + 1))

        if args.protein_asl:
            all_results = calc_rmsd(args.protein_asl, cms_model, msys_model, tr, st)
        else:
            all_results = calc_rmsd('', cms_model, msys_model, tr, st)
        print("Rep{} Protein RMSD and RMSF Analysis Done".format(index + 1))

        # Essentially this will output the RMSD data to the output array in such a way that it writes down the column
        RMSD_output[:, index] = all_results

    RMSD_row_labels = list(range(0, traj_len, 1))

    RMSD_output = np.reshape(RMSD_output, (traj_len, len(args.infiles)))

    RMSD_column_labels = []
    for i in range(len(args.infiles)):
        RMSD_column_labels.append('{} rep{} CA RMSD'.format(args.outname, i + 1))
    RMSD_column_labels.insert(0, 'Frame #')
    RMSD_column_labels.insert(1, 'Time [ns]')
    RMSD_column_labels.append('{} Average CA RMSD'.format(args.outname))

    # write out the final protein RMSD results
    RMSD_average = np.average(RMSD_output, axis=1)
    sim_time = np.around(np.linspace(0, fr.time / 1000, traj_len), 5)
    RMSD_final_output = np.column_stack((RMSD_row_labels, sim_time, RMSD_output, RMSD_average))
    RMSD_final_output = np.vstack((RMSD_column_labels, RMSD_final_output))
    with open(args.outname + '_RMSD.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(RMSD_final_output)


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)
