#!/usr/bin/env python3
"""
Parse Desmond *_nonbonded.json* files and export residue-pair interaction
energies.

Residue labels are written as '<resnum>_<chain>'.  Pair keys therefore look like

    23_A-45_B       (protein–protein)
    23_A-1_L1       (protein–ligand)

The PP / PL classification relies **only** on the ASL selections provided by
the user (`protein_asl`, `ligand_asl`), so chain anomalies no longer affect
which category a pair falls into.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import pandas as pd
import schrodinger.application.desmond.packages.topo as topo


# ──────────────────────────────────────────────────────────────────────────────
#                         per-replica energy extraction
# ──────────────────────────────────────────────────────────────────────────────
def get_pairwise_and_write_csv(
    cms_file: str,
    json_file: str,
    protein_asl: str,
    ligand_asl: str,
    output_prefix: str,
    energy_component: str,
) -> tuple[dict, dict, dict, dict]:
    """
    Extract pair-wise energies from one *_nonbonded.json* and write per-replica
    CSVs (<prefix>_pp.csv and <prefix>_pl.csv).

    Returns
    -------
    pl_mean, pl_var, pp_mean, pp_var  (four dicts, variance not σ)
    """
    _, cms = topo.read_cms(cms_file)

    # Map numeric chain indices → letters; negatives → L1, L2, …
    chain_lookup = {i: (ch.name or f"L{i}") for i, ch in enumerate(cms.chain)}

    def chain_name(raw: str | int) -> str:
        if isinstance(raw, (int, float)):
            idx = int(raw)
            return chain_lookup.get(idx, f"L{abs(idx)}")
        return str(raw) or "?"

    def fmt(resid: tuple[int, str | int]) -> str:
        resnum, ch = resid
        return f"{resnum}_{chain_name(ch)}"

    # ── read JSON ────────────────────────────────────────────────────────────
    with open(json_file) as fh:
        nb = json.load(fh)["default"]

    groups     = nb["group_ids"]                                  # [ [resnum,chain], ... ]
    pairs      = nb["results"][energy_component]["keys"]          # [ [gidA,gidB], ... ]
    pair_mean  = nb["results"][energy_component]["mean_potential"]
    pair_err   = nb["results"][energy_component]["error"]

    # Atom-level selections → group indices
    def asl_to_group_idx(asl: str) -> set[int]:
        gids = set(topo.asl2gids(cms, asl))
        resids = {(a.resnum, a.chain) for i, a in enumerate(cms.atom) if i in gids}
        return {i for i, r in enumerate(groups) if tuple(r) in resids}

    prot_idx = asl_to_group_idx(protein_asl)
    lig_idx  = asl_to_group_idx(ligand_asl)

    # containers
    pl_mean, pl_var = defaultdict(float), defaultdict(float)
    pp_mean, pp_var = defaultdict(float), defaultdict(float)

    # ── accumulate energies ─────────────────────────────────────────────────
    for (gid_a, gid_b), e_val, e_err in zip(pairs, pair_mean, pair_err):
        a_in_prot, b_in_prot = gid_a in prot_idx, gid_b in prot_idx
        a_in_lig , b_in_lig  = gid_a in lig_idx , gid_b in lig_idx

        # classify pair
        if a_in_prot and b_in_prot:           # PP
            resid_a, resid_b = map(tuple, (groups[gid_a], groups[gid_b]))
            r1, r2 = sorted((resid_a, resid_b), key=lambda r: (chain_name(r[1]), r[0]))
            key = f"{fmt(r1)}-{fmt(r2)}"
            pp_mean[key] += e_val
            pp_var[key]  += e_err**2

        elif (a_in_prot and b_in_lig) or (a_in_lig and b_in_prot):  # PL
            prot_resid = tuple(groups[gid_a]) if a_in_prot else tuple(groups[gid_b])
            lig_resid  = tuple(groups[gid_b]) if a_in_prot else tuple(groups[gid_a])
            key = f"{fmt(prot_resid)}-{fmt(lig_resid)}"
            pl_mean[key] += e_val
            pl_var[key]  += e_err**2
        # else: ignore (e.g. solvent, lipid, ligand–ligand)

    # ── write per-replica CSVs ───────────────────────────────────────────────
    #_dicts_to_df(pp_mean, pp_var, "prot_prot_pair").to_csv(f"{output_prefix}_pp.csv")
    #_dicts_to_df(pl_mean, pl_var, "prot_lig_pair").to_csv(f"{output_prefix}_pl.csv")

    return dict(pl_mean), dict(pl_var), dict(pp_mean), dict(pp_var)


# ──────────────────────────────────────────────────────────────────────────────
#                         common helper functions
# ──────────────────────────────────────────────────────────────────────────────
def _dicts_to_df(mean: dict, var: dict, idx_name: str) -> pd.DataFrame:
    """Convert mean & variance dicts to DataFrame (σ = √var)."""
    df = pd.DataFrame(
        {"mean_energy": mean,
         "error": {k: math.sqrt(v) for k, v in var.items()}}
    )
    df.index.name = idx_name
    return df


def write_summary_csv(mean: dict, var: dict, outfile: str, index_label: str) -> None:
    _dicts_to_df(mean, var, index_label).to_csv(outfile)


# ──────────────────────────────────────────────────────────────────────────────
#                               CLI & driver
# ──────────────────────────────────────────────────────────────────────────────
def cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Average replica non-bonded energies and export pair-wise "
                     "vdW or Coulomb interactions.  PP vs PL is determined by "
                     "user-specified ASL selections."))
    p.add_argument("infile", nargs="+",
                   help="One or more *_nonbonded.json files (replicas).")
    p.add_argument("-cms_file", required=True,
                   help="Path to the <job>.cms file.")
    p.add_argument("-jobname", default="nonbonded_energy",
                   help="Prefix for output files.")
    p.add_argument("-energy_component", default="nonbonded_vdw",
                   choices=["nonbonded_vdw", "nonbonded_coul"],
                   help="Energy term to extract.")
    p.add_argument("-protein_asl", default="protein",
                   help="ASL selecting all protein atoms.")
    p.add_argument("-ligand_asl", default="ligand and not a.element H",
                   help="ASL selecting all ligand atoms.")
    return p


def main(args: argparse.Namespace) -> None:
    pl_mean, pl_var = defaultdict(float), defaultdict(float)
    pp_mean, pp_var = defaultdict(float), defaultdict(float)
    n_rep = len(args.infile)

    for nb_json in args.infile:
        prefix = f"{args.jobname}_{Path(nb_json).stem}"
        _pl_m, _pl_v, _pp_m, _pp_v = get_pairwise_and_write_csv(
            cms_file=args.cms_file,
            json_file=nb_json,
            protein_asl=args.protein_asl,
            ligand_asl=args.ligand_asl,
            output_prefix=prefix,
            energy_component=args.energy_component,
        )
        # accumulate sums
        for k, v in _pl_m.items():  pl_mean[k] += v
        for k, v in _pl_v.items():  pl_var[k]  += v
        for k, v in _pp_m.items():  pp_mean[k] += v
        for k, v in _pp_v.items():  pp_var[k]  += v

    # average means
    for d in (pl_mean, pp_mean):
        for k in d: d[k] /= n_rep
    # propagate variance: σ_total = √(Σσ²) / N
    for k in pl_var: pl_var[k] = math.sqrt(pl_var[k]) / n_rep
    for k in pp_var: pp_var[k] = math.sqrt(pp_var[k]) / n_rep

    job = args.jobname
    write_summary_csv(pl_mean, pl_var, f"{job}_pl.csv", "prot_lig_pair")
    write_summary_csv(pp_mean, pp_var, f"{job}_pp.csv", "prot_prot_pair")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(cli_parser().parse_args())
