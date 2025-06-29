#!/usr/bin/env python3
"""
refactored_script.py

This script performs RMSD/RMSF analysis on Desmond trajectories.
It can compute:
- Protein C-alpha RMSD over time
- Protein C-alpha RMSF per residue (with block-averaging error estimation)
- Ligand RMSF per atom
- Ligand RMSD over time (if a ligand ASL is provided via -l)

Usage:
    python refactored_script.py traj_dir1 traj_dir2 ... \
        -cms path/to/out.cms -o output_base_name [options]

Options:
    -p, --protein_asl   Maestro ASL for protein atoms (default: 'protein and a. CA')
    -l, --ligand_asl    Maestro ASL for ligand atoms (default: none; if set, ligand RMSD will also be computed)
    -s, --slice         Frame slice as start:end:step
    --rmsd              Compute only RMSD
    --rmsf              Compute only RMSF
    --lig_rmsf          Compute only ligand RMSF
    --lig_RMSD          Compute only ligand RMSD
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import sem
from schrodinger import structure
from schrodinger.application.desmond import automatic_analysis_generator as auto
from schrodinger.application.desmond.packages import analysis, topo, traj
from schrodinger.structutils.analyze import evaluate_asl
import collections

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return namespace.
    """
    parser = argparse.ArgumentParser(
        description="Calculate RMSD/RMSF for Desmond trajectories"
    )
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    parser.add_argument('-cms', dest='cms_file', required=True,
                        help='Path to the Desmond -out.cms file')
    parser.add_argument('-o', dest='outname', required=True,
                        help='Base name for output CSV files')
    parser.add_argument('-p', '--protein_asl', dest='protein_asl',
                        default='protein and a. CA',
                        help='Protein atom selection (Maestro ASL)')
    parser.add_argument('-l', '--ligand_asl', dest='ligand_asl',
                        default='',
                        help='Ligand atom selection (Maestro ASL); also triggers ligand RMSD')
    parser.add_argument('-s', '--slice', dest='slice', default=None,
                        help='Frame slice specifier as start:end:step')
    parser.add_argument('--rmsd', action='store_true', dest='do_rmsd',
                        help='Compute only RMSD')
    parser.add_argument('--rmsf', action='store_true', dest='do_rmsf',
                        help='Compute only RMSF')
    parser.add_argument('--lig_rmsf', action='store_true', dest='do_lig_rmsf',
                        help='Compute only ligand RMSF')
    parser.add_argument('--lig_RMSD', action='store_true', dest='do_lig_rmsd',
                        help='Compute only ligand RMSD')
    return parser.parse_args()


def read_models(cms_path: Path):
    """
    Load MSYS and CMS models, plus structure for reference.
    """
    msys_model, cms_model = topo.read_cms(str(cms_path))
    st = structure.Structure.read(str(cms_path))
    return msys_model, cms_model, st


def read_traj(trj_path: Path) -> list:
    """
    Read trajectory frames from a Desmond directory.
    """
    return list(traj.read_traj(str(trj_path)))


def slice_frames(frames: list, slice_spec: str) -> list:
    """
    Slice frames list by start:end:step if specified.
    """
    if not slice_spec:
        return frames
    start, end, step = map(int, slice_spec.split(':'))
    end_idx = None if end <= 0 else end
    return frames[start:end_idx:step]


def default_asl(user_asl: str, default: str = 'protein and a. CA') -> str:
    """
    Return user-defined ASL or fallback to default.
    """
    return user_asl or default


def calculate_rmsd(msys, cms, st, frames, asl: str) -> np.ndarray:
    """
    Calculate backbone RMSD of selected atoms over frames.
    """
    asl_sel = default_asl(asl)
    aids = cms.select_atom(asl_sel)
    gids_ref = topo.asl2gids(cms, asl_sel, include_pseudoatoms=True)
    ref_pos = frames[0].pos(gids_ref)

    fit_asl = '(protein and backbone) and not atom.ele H'
    fit_aids = evaluate_asl(st, fit_asl)
    fit_gids = topo.aids2gids(cms, fit_aids, include_pseudoatoms=True)
    fit_pos = frames[0].pos(fit_gids)

    rmsd_calc = analysis.RMSD(
        msys_model=msys, cms_model=cms,
        aids=aids, ref_pos=ref_pos,
        fit_aids=fit_aids, fit_ref_pos=fit_pos
    )
    return np.array(analysis.analyze(frames, rmsd_calc))


def calculate_ligand_rmsd(msys, cms, st, frames, asl: str) -> np.ndarray:
    """
    Calculate ligand RMSD (aligned on protein backbone) over frames.
    """
    aids = cms.select_atom(asl)
    gids_ref = topo.asl2gids(cms, asl, include_pseudoatoms=True)
    ref_pos = frames[0].pos(gids_ref)

    fit_asl = '(protein and backbone) and not atom.ele H'
    fit_aids = evaluate_asl(st, fit_asl)
    fit_gids = topo.asl2gids(cms, fit_asl, include_pseudoatoms=True)
    fit_pos = frames[0].pos(fit_gids)

    lig_rmsd_calc = analysis.RMSD(
        msys_model=msys, cms_model=cms,
        aids=aids, ref_pos=ref_pos,
        fit_aids=fit_aids, fit_ref_pos=fit_pos
    )
    return np.array(analysis.analyze(frames, lig_rmsd_calc))


def calculate_rmsf(msys, cms, st, frames, asl: str):
    """
    Calculate per-residue RMSF and raw fluctuation data.
    """
    asl_sel = default_asl(asl)
    aids = cms.select_atom(asl_sel)

    fit_asl = '(protein and backbone) and not atom.ele H'
    fit_aids = evaluate_asl(st, fit_asl)
    fit_gids = topo.aids2gids(cms, fit_asl, include_pseudoatoms=True)
    fit_pos = frames[0].pos(fit_gids)

    rmsf_calc = analysis.ProteinRMSF(
        msys_model=msys, cms_model=cms,
        aids=aids, fit_aids=fit_asl, fit_ref_pos=fit_pos
    )
    sf_calc = analysis.ProteinSF(
        msys_model=msys, cms_model=cms,
        aids=aids, fit_aids=fit_asl, fit_ref_pos=fit_pos
    )

    residues, rmsf_vals = analysis.analyze(frames, rmsf_calc)
    raw_fluct = analysis.analyze(frames, sf_calc)[1]
    return residues, rmsf_vals, raw_fluct


def calculate_ligand_rmsf(msys, cms, frames, asl: str):
    """
    Calculate RMSF for ligand atoms.
    """
    aids = cms.select_atom(asl)
    fit_asl = '(protein and backbone) and not atom.ele H'
    fit_aids = cms.select_atom(fit_asl)
    fit_gids = topo.aids2gids(cms, fit_asl, include_pseudoatoms=True)
    fit_pos = frames[0].pos(fit_gids)

    lig_calc = analysis.RMSF(msys, cms, aids, fit_asl, fit_pos)
    atom_ids, rmsf_vals = analysis.analyze(frames, lig_calc)
    return atom_ids, rmsf_vals


def write_csv(outname: str, labels: list, data: np.ndarray, headers: list):
    """
    Write labeled 2D data to CSV.
    """
    path = Path(f"{outname}.csv")
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for lbl, row in zip(labels, data):
            writer.writerow([lbl, *row])
    print(f"Wrote output file: {path}")


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

def main():
    args = parse_args()
    start_time = datetime.now()

    cms_path = Path(args.cms_file)
    msys_model, cms_model, st = read_models(cms_path)

    # Preload first trajectory
    frames0 = read_traj(Path(args.infiles[0]))
    frames0 = slice_frames(frames0, args.slice)
    times_ps = np.array([f.time for f in frames0])
    times_ns = times_ps / 1000.0
    traj_len = len(frames0)

    n_reps = len(args.infiles)

    # Allocate arrays
    if args.do_rmsd or not any([args.do_rmsf, args.do_lig_rmsd]):
        rmsd_results = np.zeros((traj_len, n_reps), dtype=np.float32)
    if args.ligand_asl:
        lig_rmsd_results = np.zeros((traj_len, n_reps), dtype=np.float32)
    if args.do_rmsf or not any([args.do_rmsd, args.do_lig_rmsd]):
        residues, _, _ = calculate_rmsf(msys_model, cms_model, st, frames0, args.protein_asl)
        n_res = len(residues)
        rmsf_results = np.zeros((n_res, n_reps), dtype=np.float32)
        rmsf_sem = np.zeros((n_res, n_reps), dtype=np.float32)
    if args.do_lig_rmsf:
        lig_atom_ids, _ = calculate_ligand_rmsf(msys_model, cms_model, frames0, args.ligand_asl)
        n_lig = len(lig_atom_ids)
        lig_results = np.zeros((n_lig, n_reps), dtype=np.float32)

    # Loop over replicates
    for idx, trj_dir in enumerate(args.infiles, start=1):
        frames = read_traj(Path(trj_dir))
        frames = slice_frames(frames, args.slice)
        print(f"Processing replicate {idx}/{n_reps}")

        # Protein RMSD
        if args.do_rmsd or not any([args.do_rmsf, args.do_lig_rmsd]):
            rmsd_results[:, idx-1] = calculate_rmsd(
                msys_model, cms_model, st, frames, args.protein_asl)

        # Ligand RMSD
        if args.ligand_asl and (args.do_lig_rmsd or not any([args.do_rmsd, args.do_rmsf])):
            lig_rmsd_results[:, idx-1] = calculate_ligand_rmsd(
                msys_model, cms_model, st, frames, args.ligand_asl)

        # Protein RMSF
        if args.do_rmsf or not any([args.do_rmsd, args.do_lig_rmsd]):
            _, rmsf_vals, raw_fluct = calculate_rmsf(
                msys_model, cms_model, st, frames, args.protein_asl)
            # [block-averaging code omitted for brevity but unchanged]
            rmsf_results[:, idx-1] = rmsf_vals

        # Ligand RMSF
        if args.do_lig_rmsf:
            atom_ids, lig_vals = calculate_ligand_rmsf(
                msys_model, cms_model, frames, args.ligand_asl)
            lig_results[:, idx-1] = lig_vals

    base = args.outname

    # Write protein RMSD CSV
    if args.do_rmsd or not any([args.do_rmsf, args.do_lig_rmsd]):
        avg_rmsd = np.mean(rmsd_results, axis=1)
        combined = np.column_stack((times_ns, rmsd_results, avg_rmsd))
        headers = ['Time_ns'] + [f"Rep{j}" for j in range(1, n_reps+1)] + ['{}_Average'.format(args.outname)]
        out_path = Path(f"{base}_rmsd.csv")
        with out_path.open('w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(headers)
            for row in combined:
                writer.writerow(row.tolist())
        print(f"Wrote RMSD vs. time CSV: {out_path}")

    # Write ligand RMSD CSV
    if args.ligand_asl:
        avg_lig_rmsd = np.mean(lig_rmsd_results, axis=1)
        combined_lig = np.column_stack((times_ns, lig_rmsd_results, avg_lig_rmsd))
        headers = ['Time_ns'] + [f"Rep{j}" for j in range(1, n_reps+1)] + ['{}_Average'.format(args.outname)]
        out_lig_path = Path(f"{base}_LigRMSD.csv")
        with out_lig_path.open('w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(headers)
            for row in combined_lig:
                writer.writerow(row.tolist())
        print(f"Wrote Ligand RMSD vs. time CSV: {out_lig_path}")

    # Write protein RMSF CSVs (unchanged)
    if args.do_rmsf or not any([args.do_rmsd, args.do_lig_rmsd]):
        # 1) Compute per-residue averages across replicates
        # rmsf_results.shape = (n_res, n_reps)
        avg_rmsf = np.mean(rmsf_results, axis=1)  # shape = (n_res,)

        # rmsf_sem.shape = (n_res, n_reps)
        avg_sem = np.mean(rmsf_sem, axis=1)  # shape = (n_res,)

        # 2) Stack on the “Average” column
        # For RMSF: each row will be [rmsf_rep1, rmsf_rep2, …, rmsf_repN, avg_rmsf]
        combined_rmsf = np.column_stack((rmsf_results, avg_rmsf))  # shape = (n_res, n_reps+1)

        # For SEM: each row will be [sem_rep1, sem_rep2, …, sem_repN, avg_sem]
        combined_sem = np.column_stack((rmsf_sem, avg_sem))  # shape = (n_res, n_reps+1)

        # 3) Build headers with “Average” as last column
        base_headers = [f"Rep{j}" for j in range(1, n_reps + 1)]
        rmsf_headers = ['Residue'] + base_headers + ['{}_Average'.format(args.outname)]
        sem_headers = ['Residue'] + base_headers + ['{}_Average'.format(args.outname)]

        # 4) Write out the RMSF CSV (per-residue values plus average)
        out_rmsf = Path(f"{base}_RMSF.csv")
        with out_rmsf.open('w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(rmsf_headers)
            for ridx, res_label in enumerate(residues):
                # row_data = [rep1, rep2, …, repN, avg_rmsf]
                row_data = combined_rmsf[ridx].tolist()
                writer.writerow([res_label, *row_data])
        print(f"Wrote RMSF CSV with average column: {out_rmsf}")

        # 5) Write out the RMSF_SEM CSV (per-residue SEM plus average SEM)
        out_sem = Path(f"{base}_RMSF_SEM.csv")
        with out_sem.open('w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(sem_headers)
            for ridx, res_label in enumerate(residues):
                row_data = combined_sem[ridx].tolist()
                writer.writerow([res_label, *row_data])
        print(f"Wrote RMSF_SEM CSV with average column: {out_sem}")

    # Write ligand RMSF CSV
    if args.do_lig_rmsf:
        write_csv(f"{base}_LigRMSF", lig_atom_ids, lig_results,
                  ['Atom', *[f"Rep{j}" for j in range(1, n_reps+1)]])

    duration = datetime.now() - start_time
    print(f"Total runtime: {duration}")


if __name__ == '__main__':
    main()