import matplotlib
matplotlib.use('Agg')
from schrodinger.application.desmond.packages import traj, topo, analysis
import csv
from datetime import datetime
from schrodinger import structure
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.application.desmond import automatic_analysis_generator as auto
import matplotlib.pyplot as plt
import numpy as np
import argparse
import collections
from scipy.stats import sem


# get the time the script starts running - This is used so we can know how long things take to run
startTime = datetime.now()


def read_cms_file(cms_file):
    """
    This function will read the input cms_file
    :param str cms_file: file path for the input cms file
    :returns msys_model and cms_model
    """

    # Load the cms_model
    msys_model, cms_model = topo.read_cms(cms_file)
    # return the msys_model and cms_model
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


def calc_rmsd_rmsf(optional_protein_asl, cms_model, msys_model, trj_out, st):
    """
    Calculates the c-alpha RMSD and RMSF
    :param str optional_protein_asl:
    :param cms_model
    :param msys_model
    :param trj_out
    :param st
    :returns results[0][0]) = this is the residue info
    :returns (results[0][1]) = this is RMSF
    :returns (results[1]) = this is RMSD
    """

    # This is the ASL to use to calculate the RMSD and RMSF
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

    # calculate RMSD
    # aids =
    # ref_pos =
    # fit_aids =
    # fit_ref_pos =
    rmsd_analysis = analysis.RMSD(msys_model=msys_model, cms_model=cms_model, aids=aids, ref_pos=ref_pos,
                                  fit_aids=fit_aids, fit_ref_pos=fit_ref_pos)

    # aids_rmsf = evaluate_asl(st, rmsf_aids_selection)
    # TODO give the user the option to calculate the RMSF for something else other than c alpha atoms?
    aids_rmsf = cms_model.select_atom(ASL_calc)

    if ligand[0] == None:
        # rmsf_fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H) and not (({})))'.
        #                              format(ligand[1]))
        rmsf_fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H))')
    else:
        # rmsf_fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H))')
        rmsf_fit_aids = evaluate_asl(st, '(((protein) and backbone) and not (atom.ele H) and not (({})))'.
                                     format(ligand[1]))

    rmsf_fit_gids = topo.aids2gids(cms_model, rmsf_fit_aids, include_pseudoatoms=False)
    rmsf_fit_ref_pos = trj_out[0].pos(rmsf_fit_gids)

    # calculate RMSF
    # aids =
    # fit_aids =
    # fit_ref_pos =
    rmsf_analysis = analysis.ProteinRMSF(msys_model=msys_model, cms_model=cms_model, aids=aids_rmsf,
                                         fit_aids=rmsf_fit_aids, fit_ref_pos=rmsf_fit_ref_pos)


    per_resi_per_frame_SF = analysis.ProteinSF(msys_model=msys_model, cms_model=cms_model, aids=aids_rmsf,
                                         fit_aids=rmsf_fit_aids, fit_ref_pos=rmsf_fit_ref_pos)

    # (results[0][0]) = this is the residue info
    # (results[0][1]) = this is RMSF
    # (results[1]) = this is RMSD
    results = (analysis.analyze(trj_out, rmsf_analysis, rmsd_analysis,
                                progress_feedback=analysis.progress_report_frame_number))
    results_SF = (analysis.analyze(trj_out,per_resi_per_frame_SF,
                                   progress_feedback=analysis.progress_report_frame_number))

    return results, results_SF


# See (https://pyblock.readthedocs.io/_/downloads/en/stable/pdf/) for block averaging
def reblock(data, rowvar=1, ddof=None, weights=None):
    '''Blocking analysis of correlated data.

Repeatedly average neighbouring data points in order to remove the effect of
serial correlation on the estimate of the standard error of a data set, as
described by Flyvbjerg and Petersen [Flyvbjerg]_.  The standard error is constant
(within error bars) once the correlation has been removed.

If a weighting is provided then the weighted variance and standard error of
each variable is calculated, as described in [Pozzi]_. Bessel correction is
obtained using the "effective sample size" from [Madansky]_.

.. default-role:: math

Parameters
----------
data : :class:`numpy.ndarray`
    1D or 2D array containing multiple variables and data points.  See ``rowvar``.
rowvar : int
    If ``rowvar`` is non-zero (default) then each row represents a variable and
    each column a data point per variable.  Otherwise the relationship is
    swapped.  Only used if data is a 2D array.
ddof : int
    If not ``None``, then the standard error and covariance are normalised by
    `(N - \\text{ddof})`, where `N` is the number of data points per variable.
    Otherwise, the numpy default is used (i.e. `(N - 1)`).
weights : :class:`numpy.array`
    A 1D weighting of the data to be reblocked. For multidimensional data an
    identical weighting is applied to the data for each variable.

Returns
-------
block_info : :class:`list` of :func:`collections.namedtuple`
    Statistics from each reblocking iteration.  Each tuple contains:

        block : int
            blocking iteration.  Each iteration successively averages neighbouring
            pairs of data points.  The final data point is discarded if the number
            of data points is odd.
        ndata: int
            number of data points in the blocking iteration.
        mean : :class:`numpy.ndarray`
            mean of each variable in the data set.
        cov : :class:`numpy.ndarray`
            covariance matrix.
        std_err : :class:`numpy.ndarray`
            standard error of each variable.
        std_err_err : :class:`numpy.ndarray`
            an estimate of the error in the standard error, assuming a Gaussian
            distribution.

References
----------
.. [Flyvbjerg]  "Error estimates on averages of correlated data", H. Flyvbjerg,
   H.G. Petersen, J. Chem. Phys. 91, 461 (1989).
.. [Pozzi]  "Exponential smoothing weighted correlations", F. Pozzi, T. Matteo,
   T. Aste, Eur. Phys. J. B. 85, 175 (2012).
.. [Madansky]  "An Analysis of WinCross, SPSS, and Mentor Procedures for
   Estimating the Variance of a Weighted Mean", A. Madansky, H. G. B. Alexander,
   www.analyticalgroup.com/download/weighted_variance.pdf
'''

    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")
    if ddof is None:
        ddof = 1

    if data.ndim > 2:
        raise RuntimeError("do not understand how to reblock in more than two dimensions")

    if data.ndim == 1 or data.shape[0] == 1:
        rowvar = 1
        axis = 0
        nvar = 1
    elif rowvar:
        nvar = data.shape[0]
        axis = 1
    else:
        nvar = data.shape[1]
        axis = 0

    if weights is not None:
        if weights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional weights")
        if weights.shape[0] != data.shape[axis]:
            raise RuntimeError("incompatible numbers of weights and samples")
        if np.any(weights < 0):
            raise RuntimeError("cannot handle negative weights")

    iblock = 0
    stats = []
    block_tuple_fields = 'block ndata mean cov std_err std_err_err'.split()
    block_tuple = collections.namedtuple('BlockTuple', block_tuple_fields)
    while data.shape[axis] >= 2:

        data_len = data.shape[axis]

        if weights is None:
            nsamp = data_len
            mean = np.array(np.mean(data, axis=axis))
            cov = np.cov(data, rowvar=rowvar, ddof=ddof)
        else:
            mean, tot_weight = np.average(data,
                                             axis=axis,
                                             weights=weights,
                                             returned=True)
            mean = np.array(mean)
            if nvar > 1:
                tot_weight = tot_weight[0]
            norm_wts = weights/tot_weight
            nsamp = 1.0/np.sum(norm_wts*norm_wts)
            bessel = nsamp/(nsamp - ddof)
            if data.ndim == 1:
                ds = data - mean
                cov = bessel*np.sum(norm_wts*ds*ds)
            else:
                ds = np.empty((nvar, data_len))
                for i in range(nvar):
                    d = data[i, :] if rowvar else data[:, i]
                    ds[i] = d - mean[i]
                cov = np.zeros((nvar, nvar))
                for i in range(nvar):
                    for j in range(i, nvar):
                        cov[i, j] = bessel*np.sum(norm_wts*ds[i]*ds[j])
            if nvar > 1:
                cov = cov + cov.T - np.diag(cov.diagonal())

        if cov.ndim < 2:
            std_err = np.array(np.sqrt(cov/nsamp))
        else:
            std_err = np.sqrt(cov.diagonal()/nsamp)
        std_err_err = np.array(std_err/(np.sqrt(2*(nsamp - ddof))))
        stats.append(
            block_tuple(iblock, data_len, mean, cov, std_err, std_err_err)
        )

        # last even-indexed value (ignore the odd one, if relevant)
        half = int(data.shape[axis]/2)
        last = 2*half
        if weights is None:
            if data.ndim == 1 or not rowvar:
                data = (data[:last:2] + data[1:last:2])/2
            else:
                data = (data[:,:last:2] + data[:,1:last:2])/2
        else:
            weights = norm_wts[:last:2] + norm_wts[1:last:2]
            if data.ndim == 1:
                wt_data = data[:last]*norm_wts[:last]
                data = (wt_data[::2] + wt_data[1::2])/weights
            elif rowvar:
                wt_data = data[:,:last]*norm_wts[:last]
                data = (wt_data[:,::2] + wt_data[:,1::2])/weights
            else:
                idxs = 'ij,i->ij'
                wt_data = np.einsum(idxs, data[:last], norm_wts[:last])
                summed_wt_data = wt_data[::2] + wt_data[1::2]
                data = np.einsum(idxs, summed_wt_data, 1.0/weights)

        iblock += 1

    return stats


# See (https://pyblock.readthedocs.io/_/downloads/en/stable/pdf/) for block averaging
# Find the optimal number of blocks to divide the data into
def find_optimal_block(ndata, stats):
    '''Find the optimal block length from a reblocking calculation.

Inspect a reblocking calculation and find the block length which minimises the
stochastic error and removes the effect of correlation from the data set.  This
follows the procedures detailed by [Wolff]_ and [Lee]_ et al.

.. default-role:: math

Parameters
----------
ndata : int
    number of data points ('observations') in the data set.
stats : list of tuples
    statistics in the format as returned by :func:`pyblock.blocking.reblock`.

Returns
-------
list of int
    the optimal block index for each variable (i.e. the first block index in
    which the correlation has been removed).  If NaN, then the statistics
    provided were not sufficient to estimate the correlation length and more
    data should be collected.

Notes
-----
[Wolff]_ (Eq 47) and [Lee]_ et al. (Eq 14) give the optimal block size to be

.. math::

    B^3 = 2 n n_{\\text{corr}}^2

where `n` is the number of data points in the data set, `B` is the number of
data points in each 'block' (ie the data set has been divided into `n/B`
contiguous blocks) and `n_{\\text{corr}}`.
[todo] - describe n_corr.
Following the scheme proposed by [Lee]_ et al., we hence look for the largest
block size which satisfies

.. math::

    B^3 >= 2 n n_{\\text{corr}}^2.

From Eq 13 in [Lee]_ et al. (which they cast in terms of the variance):

.. math::

    n_{\\text{err}} SE = SE_{\\text{true}}

where the 'error factor', `n_{\\text{err}}`, is the square root of the
estimated correlation length,  `SE` is the standard error of the data set and
`SE_{\\text{true}}` is the true standard error once the correlation length has
been taken into account.  Hence the condition becomes:

.. math::

    B^3 >= 2 n (SE(B) / SE(0))^4

where `SE(B)` is the estimate of the standard error of the data divided in
blocks of size `B`.

I am grateful to Will Vigor for discussions and the initial implementation.

References
----------
.. [Wolff] "Monte Carlo errors with less errors", U. Wolff, Comput. Phys. Commun.
       156, 143 (2004) and arXiv:hep-lat/0306017.
.. [Lee] "Strategies for improving the efficiency of quantum Monte Carlo
       calculations", R.M. Lee, G.J. Conduit, N. Nemec, P. Lopez Rios,
       N.D.  Drummond, Phys. Rev. E. 83, 066706 (2011).
'''

    # Get the number of variables by looking at the number of means calculated
    # in the first stats entry.
    nvariables = stats[0][2].size
    optimal_block = [float('NaN')]*nvariables
    # If the data was just of a single variable, then the numpy arrays returned
    # by blocking are all 0-dimensions.  Make sure they're 1-D so we can use
    # enumerate safely (just to keep the code short).
    std_err_first = np.array(stats[0][4], ndmin=1)
    for (iblock, data_len, mean, cov, std_err, std_err_err) in reversed(stats):
        # 2**iblock data points per block.
        B3 = 2**(3*iblock)
        std_err = np.array(std_err, ndmin=1)
        for (i, var_std_err) in enumerate(std_err):
            if B3 > 2*ndata*(var_std_err/std_err_first[i])**4:
                optimal_block[i] = iblock

    return optimal_block


# Calculate the Ligand RMSF
def calc_lig_rmsf(optional_ligand_asl, cms_model, msys_model, trj_out, st):
    # This evaluates if the user inputs a ligand ASL. If yes, then use the user input
    # if the user did not input a ligand ASL, then use the auto.getLigand to get the ligand
    aids = cms_model.select_atom(optional_ligand_asl)

    # we're fitting on the protein
    # TODO do we need to fit on the same thing as the RMSD/RMSF function
    fit_ASL = '(((protein) and backbone) and not (atom.ele H))'
    fit_aids = cms_model.select_atom(fit_ASL)

    # get the first frame of the trajectory
    fr = trj_out[0]

    fit_gids = topo.aids2gids(cms_model, fit_aids, include_pseudoatoms=False)
    fit_ref_pos = fr.pos(fit_gids)

    lig_RMSF_analysis = analysis.RMSF(msys_model, cms_model, aids, fit_aids, fit_ref_pos)

    results = analysis.analyze(trj_out, lig_RMSF_analysis,
                               progress_feedback=analysis.progress_report_frame_number(1000))

    # lig_aids = results[0]
    # lig_rmsf = results[1]
    return aids, results


def write_csv(cms_model, RMSD_output, protein_RMSF_output, traj_len, RMSF_row_labels, fr, errors,
              atom_ids, lig_RMSF_output):
    RMSD_row_labels = list(range(0, traj_len, 1))

    RMSF_row_labels_ASL_calc = 'protein and a. CA'
    RMSF_row_labels_aids = cms_model.select_atom(RMSF_row_labels_ASL_calc)

    RMSD_output = np.reshape(RMSD_output, (traj_len, len(args.infiles)))
    protein_RMSF_output = np.reshape(protein_RMSF_output, (len(RMSF_row_labels_aids), len(args.infiles)))
    # TODO deal with no ligands
    # lig_RMSF_output = np.reshape(lig_RMSF_output, (len(ligaids), len(args.infiles)))

    RMSD_column_labels = []
    for i in range(len(args.infiles)):
        RMSD_column_labels.append('{} rep{} CA RMSD'.format(args.outname, i+1))
    RMSD_column_labels.insert(0, 'Frame #')
    RMSD_column_labels.insert(1, 'Time [ns]')
    RMSD_column_labels.append('{} Average CA RMSD'.format(args.outname))

    RMSF_column_labels = []
    for i in range(len(args.infiles)):
        RMSF_column_labels.append('{} rep{} CA RMSF'.format(args.outname, i+1))
    RMSF_column_labels.insert(0, 'Residue ID')
    RMSF_column_labels.append('{} Average CA RMSF'.format(args.outname))
    RMSF_column_labels.append ('{} SEM CA RMSF'.format (args.outname))

    # write out the final protein RMSD results
    RMSD_average = np.average(RMSD_output, axis=1)
    sim_time = np.around(np.linspace(0, fr.time/1000, traj_len), 5)
    RMSD_final_output = np.column_stack((RMSD_row_labels, sim_time, RMSD_output, RMSD_average))
    RMSD_final_output = np.vstack((RMSD_column_labels, RMSD_final_output))
    with open(args.outname + '_RMSD.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(RMSD_final_output)

    # write out the protein RMSF results
    RMSF_average = np.average(protein_RMSF_output, axis=1)
    RMSF_SEM = np.average(errors, axis=1)

    graph_RMSF_w_error(RMSF_row_labels, RMSF_average, RMSF_SEM)

    RMSF_final_output = np.column_stack((RMSF_row_labels, protein_RMSF_output, RMSF_average, RMSF_SEM))
    RMSF_final_output = np.vstack((RMSF_column_labels, RMSF_final_output))
    with open(args.outname + '_RMSF.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(RMSF_final_output)


    if atom_ids != '':
        lig_RMSF_column_labels = []
        for i in range(len(args.infiles)):
            lig_RMSF_column_labels.append('{} rep{} Ligand RMSF'.format(args.outname, i + 1))

        lig_RMSF_column_labels.insert(0, 'Ligand Atoms')
        lig_RMSF_column_labels.append('{} Average Ligand RMSF'.format(args.outname))

        lig_RMSF_average = np.average(lig_RMSF_output, axis=1)

        lig_RMSF_final_output = np.column_stack((atom_ids, lig_RMSF_output, lig_RMSF_average))
        lig_RMSF_final_output = np.vstack((lig_RMSF_column_labels, lig_RMSF_final_output))
        with open(args.outname + '_lig_RMSF.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(lig_RMSF_final_output)


def graph_data_no_lig(RMSD_output, protein_RMSF_output, traj_len, RMSF_row_labels):
    # for RMSD, the row labels will be the trj length
    RMSD_row_labels = list(range(0, traj_len, 1))

    #TODO add ligand RMSF plotting
    counter = 0
    for i, j in zip(RMSD_output.T, protein_RMSF_output.T):
        # make graphs
        # first subplot is RMSD
        plt.subplot(211)
        plt.subplots_adjust(hspace=.5)
        plt.plot(RMSD_row_labels, i, color='black', linewidth=1)
        plt.xlabel('Frame Number')
        plt.ylabel(r"RMSD ($\AA$)")
        plt.xlim((0, traj_len))
        # second subplot is protein RMSF
        plt.subplot(212)
        plt.plot(range(len(RMSF_row_labels)), j, color='black', linewidth=1)
        plt.xlabel('Residue Number')
        plt.ylabel(r"RMSF ($\AA$)")
        q = len(RMSF_row_labels)
        plt.xlim((0, q))
        # third subplot is ligand RMSF
        # plt.subplot(313)
        # plt.plot(range(len(lig_rmsf_aids)), lig_rmsf_results, color='black', linewidth=2)
        # plt.xlabel('Ligand Atom')
        # plt.ylabel(r"RMSF ($\AA$)")
        # plt.xlim((0, len(lig_rmsf_aids)))
        plt.savefig("{}_rep{}_Results.png".format(args.outname, counter + 1), dpi=300, bbox_inches='tight')
        counter += 1
        plt.close()


def graph_data_with_lig(RMSD_output, protein_RMSF_output, traj_len, RMSF_row_labels, lig_rmsf_aids, lig_RMSF_output):
    # for RMSD, the row labels will be the trj length
    RMSD_row_labels = list(range(0, traj_len, 1))

    counter = 0
    for i, j, k in zip(RMSD_output.T, protein_RMSF_output.T, lig_RMSF_output.T):
        # make graphs
        # first subplot is RMSD
        plt.subplot(311)
        plt.subplots_adjust(hspace=.5)
        plt.plot(RMSD_row_labels, i, color='black', linewidth=1)
        plt.xlabel('Frame Number')
        plt.ylabel(r"RMSD ($\AA$)")
        plt.xlim((0, traj_len))
        # second subplot is protein RMSF
        plt.subplot(312)
        plt.plot(range(len(RMSF_row_labels)), j, color='black', linewidth=1)
        plt.xlabel('Residue Number')
        plt.ylabel(r"RMSF ($\AA$)")
        q = len(RMSF_row_labels)
        plt.xlim((0, q))
        # third subplot is ligand RMSF
        plt.subplot(313)
        plt.plot(range(len(lig_rmsf_aids)), k, color='black', linewidth=2)
        plt.xlabel('Ligand Atom')
        plt.ylabel(r"RMSF ($\AA$)")
        plt.xlim((0, len(lig_rmsf_aids)))
        plt.savefig("{}_rep{}_Results.png".format(args.outname, counter + 1), dpi=300, bbox_inches='tight')
        counter += 1
        plt.close()


def graph_RMSF_w_error(RMSF_row_labels, RMSF_average, RMSF_SEM):
    # plt.plot(range(len(RMSF_row_labels)), RMSF_average, color='black', linewidth=1)
    plt.errorbar(range(len(RMSF_row_labels)), RMSF_average, yerr=RMSF_SEM, ecolor='red', elinewidth=0.25,
                 color='black', linewidth=0.5)
    plt.xlabel('Residue Number')
    plt.ylabel(r"RMSF ($\AA$)")
    q = len(RMSF_row_labels)
    plt.xlim((0, q))
    plt.savefig("{}_RMSF_w_SEM.png".format(args.outname), dpi=300, bbox_inches='tight')
    plt.close()


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
    parser.add_argument('-s',
                        help="Use every nth frame of the trajectory. Default is 1 (i.e., use every frame in the trj",
                        type=int,
                        default=1)
    return parser


def main(args):
    # Call the read_cms_file function, pass to it the *-out.cms file (script input)
    # Assign the 2 outputs the names msys_model and cms_model
    msys_model, cms_model = read_cms_file(args.cms_file)

    st = structure.Structure.read(args.cms_file)

    # Load the first trj so you can get the length of the trajectory
    # don't need to load anything else bc all input trjs should be the same length
    tr = read_trajectory(args.infiles[0])
    if args.s:
        tr = list(tr[i] for i in range(0, len(tr), args.s))

    # initialize the output arrays
    RMSD_output = np.zeros((len(tr), len(args.infiles)), dtype=np.float16)
    protein_RMSF_output = np.zeros((len(cms_model.select_atom(args.protein_asl)), len(args.infiles)), dtype=np.float16)

    # delete to manage memory
    del tr

    if args.ligand_asl != '':
        lig_aids = cms_model.select_atom(args.ligand_asl)

        # #TODO fix lig_RMSF_output
        # lig_RMSF_output = []
        # lig_RMSF_output = np.asarray(lig_RMSF_output)
        lig_RMSF_output = np.zeros((len(lig_aids), len(args.infiles)))

    #initialize the RMSF SEM output arrays
    RMSF_SEM_output = np.zeros((len(cms_model.select_atom('a. CA')), len(args.infiles)))
    # RMSF_SEM_output = []

    traj_len = 0

    # iterate over the input trajectories
    for index, trajectory in enumerate(args.infiles):
        tr = read_trajectory(trajectory)
        print (len(tr))

        # if the user wants to take strides, this will do that for them
        if args.s:
            tr = list(tr[i] for i in range(0, len(tr), args.s))

        traj_len = len(tr)
        fr = tr[-1]
        print("Done loading trajectory {}".format(index + 1))

        if args.protein_asl:
            all_results, proteinSF = calc_rmsd_rmsf(args.protein_asl, cms_model, msys_model, tr, st)
        else:
            all_results, proteinSF = calc_rmsd_rmsf('', cms_model, msys_model, tr, st)
        print("Rep{} Protein RMSD and RMSF Analysis Done".format(index + 1))

        # Essentially this will output the RMSD data to the output array in such a way that it writes down the column
        RMSD_output[:, index] = all_results[1]
        # output protein RMSF data to the output array
        # protein_RMSF_output = np.append(protein_RMSF_output, all_results[0][1], axis=0)
        protein_RMSF_output[:, index] = all_results[0][1]

        # Do block averaging
        blocking_all = []
        for i in proteinSF[1].T:
            blocking = reblock(i)
            blocking_all.append(blocking)

        optimal_block_num = []
        for j in blocking_all:
            optimal = find_optimal_block(ndata=traj_len, stats=j)
            optimal_block_num.append(optimal)

        optimal_block_num = [item for sublist in optimal_block_num for item in sublist]

        # most common block number
        most_common_number = max(set(optimal_block_num), key=optimal_block_num.count)

        per_rep_SEM = []

        # for every residue, in proteinSF divide that list into the optimal number of blocks
        for resi in proteinSF[1].T:
            divided_list = np.array_split(np.asarray(resi), most_common_number)

            # sem list
            sem_list = []
            for i in divided_list:
                sem_list.append(sem(i))

            per_rep_SEM.append(sem(sem_list))

        RMSF_SEM_output[:,index] = per_rep_SEM

        ###################################
        # calculate ligand RMSF if needed #
        ###################################

        # If the user, inputs a ligand ASL, use that
        if args.ligand_asl != '':
            # Ligand RMSF analysis with provided ligand asl
            lig_rmsf_aids, lig_rmsf_results = calc_lig_rmsf(args.ligand_asl, cms_model, msys_model, tr, st)
            lig_RMSF_output[:, index] = lig_rmsf_results
            print("Rep{} Ligand RMSF Analysis Done".format(index + 1))
        else:
            continue

        del tr
        all_results, proteinSF = [], []

    ###########################
    # Output data to CSV file #
    ###########################
    RMSD_output = np.around(RMSD_output, decimals=3)
    # write_csv(cms_model, RMSD_output, protein_RMSF_output, traj_len, all_results[0][0], fr, RMSF_SEM_output)

    if args.ligand_asl != '':
        for i in range(len(lig_rmsf_aids)):
            # to get the info of the pairs
            atoms = [a for a in cms_model.atom]
            atom_ids = []

            counter = 0
            while counter < len(lig_rmsf_aids):
                a = atoms[lig_rmsf_aids[counter] - 1]
                chain = a.chain.strip()
                resnum = a.resnum
                resname = a.pdbres.strip()
                atom_name = a.pdbname.strip()
                atom_ids.append('{} {} {} {}'.format(chain, resnum, resname, atom_name))
                counter = counter + 1

        write_csv(cms_model, RMSD_output, protein_RMSF_output, traj_len, all_results[0][0], fr, RMSF_SEM_output,
                  atom_ids, lig_RMSF_output)
    else:
        write_csv(cms_model, RMSD_output, protein_RMSF_output, traj_len, all_results[0][0], fr, RMSF_SEM_output, '', '')

    #################
    # Plot the Data #
    #################
    if args.ligand_asl != '':
        graph_data_with_lig(RMSD_output, protein_RMSF_output, traj_len, all_results[0][0],
                            lig_rmsf_aids, lig_RMSF_output)
    else:
        graph_data_no_lig(RMSD_output, protein_RMSF_output, traj_len, all_results[0][0])


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    main(args)
    print(datetime.now() - startTime)
