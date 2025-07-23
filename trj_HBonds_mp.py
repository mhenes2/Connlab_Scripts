import multiprocessing as mp
# Use fork to support multiprocessing alongside Qt
mp.set_start_method('fork', force=True)

from schrodinger.application.desmond.packages import traj, topo, analysis
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import math
from datetime import datetime

# Global models for worker processes
_g_msys_model = None
_g_cms_model = None

def init_worker(cms_file):
    """
    Initialize global models in each worker process.
    """
    global _g_msys_model, _g_cms_model
    _g_msys_model, _g_cms_model = topo.read_cms(cms_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interaction analyses leveraging all CPU cores via multiprocessing"
    )
    parser.add_argument('infiles', nargs='+',
                        help='Desmond trajectory directories')
    parser.add_argument('-cms', dest='cms_file', required=True,
                        help='Path to the Desmond -out.cms file')
    parser.add_argument('-o', dest='outname', required=True,
                        help='Base name for output CSV files')
    parser.add_argument('-protein_asl', '--protein_asl', dest='protein_asl',
                        default='((chain.name A) OR (chain.name B) OR (chain.name C))',
                        help='Protein atom selection (Maestro ASL)')
    parser.add_argument('-combined_asl', '--combined_asl', dest='combined_asl',
                        default='((chain.name A) OR (chain.name B) OR (chain.name C) OR (chain.name D) OR (chain.name E) OR (chain.name F) OR (chain.name G) OR (chain.name H) OR (chain.name I))',
                        help='Combined protein-ligand atom selection (Maestro ASL)')
    parser.add_argument('-ligand_asl', '--ligand_asl', dest='ligand_asl',
                        default='((chain.name D) OR (chain.name E) OR (chain.name F) OR (chain.name G) OR (chain.name H) OR (chain.name I))',
                        help='Ligand atom selection (Maestro ASL)')
    parser.add_argument('-workers', type=int, default=mp.cpu_count(),
                        help='Number of worker processes for parallel analysis')
    return parser.parse_args()


def analyze_chunk(args_tuple):
    """
    Worker function: process a chunk of frames from a trajectory directory.
    args_tuple = (trj_dir, asl, start, end)
    Returns dict_of_dicts.
    """
    trj_dir, asl, start, end = args_tuple
    frames = []
    for idx, frame in enumerate(traj.read_traj(trj_dir)):
        if idx < start:
            continue
        if idx >= end:
            break
        frames.append(frame)
    inter = analysis.ProtProtInter(_g_msys_model, _g_cms_model, asl=str(asl))
    return analysis.analyze(frames, inter)


def process_analyze(infiles, asl, cms_file, workers):
    """
    Split all trajectories into frame chunks and analyze each in parallel.
    Combines dict_of_dicts results across all chunks.
    """
    # First, determine frame counts per file
    file_lengths = {}
    total_frames = 0
    for d in infiles:
        count = sum(1 for _ in traj.read_traj(d))
        file_lengths[d] = count
        total_frames += count

    # Build tasks: one chunk per worker across each file
    tasks = []
    for trj_dir, length in file_lengths.items():
        if length == 0:
            continue
        chunk_size = math.ceil(length / workers)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            tasks.append((trj_dir, asl, start, end))

    # Execute in parallel
    combined = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(cms_file,)
    ) as executor:
        for part in executor.map(analyze_chunk, tasks):
            for interaction_type, inner in part.items():
                combined.setdefault(interaction_type, Counter())
                for pair, cnt in inner.items():
                    combined[interaction_type][pair] += cnt

    # Convert Counters to dicts
    return combined, total_frames


def dicts_to_csv_pandas(dict_of_dicts, interaction, outname, trj_length):
    if not trj_length:
        raise ValueError("trj_length must be provided and non-zero")
    for main_key, inner_dict in dict_of_dicts.items():
        df = pd.DataFrame(inner_dict.items(), columns=['pairs', 'frequency'])
        df['pairs'] = df['pairs'].apply(lambda tup: f"{tup[0]} - {tup[1]}")
        df['frequency'] = df['frequency'] / trj_length * 100
        filename = f"{outname}_{interaction}_{main_key}.csv"
        df.to_csv(filename, index=False, float_format="%.2f")


def filter_protein_ligand(pl_dicts, ll_dicts, pp_dicts):
    """
    Remove overlapping interaction pairs from protein-ligand results.

    Excludes any pair present in ligand-ligand OR protein-protein results.

    Parameters:
        pl_dicts (dict): mapping interaction_type -> {pair: count}
        ll_dicts (dict): mapping interaction_type -> {pair: count}
        pp_dicts (dict): mapping interaction_type -> {pair: count}

    Returns:
        dict: filtered protein-ligand dict_of_dicts with overlaps removed.
    """
    filtered = {}
    for interaction_type, pl_inner in pl_dicts.items():
        # Get overlapping pairs from both ll and pp
        ll_inner = ll_dicts.get(interaction_type, {})
        pp_inner = pp_dicts.get(interaction_type, {})
        # Keep only pairs unique to protein-ligand
        filtered_inner = {}
        for pair, count in pl_inner.items():
            if pair not in ll_inner and pair not in pp_inner:
                filtered_inner[pair] = count
        filtered[interaction_type] = filtered_inner
    return filtered


def main():
    args = parse_args()
    start_time = datetime.now()

    # Analyze protein-protein
    pp_results, total_frames = process_analyze(
        args.infiles,
        args.protein_asl,
        args.cms_file,
        args.workers
    )
    dicts_to_csv_pandas(pp_results, 'protein-protein', args.outname, total_frames)

    # Analyze protein-ligand and ligand-ligand
    pl_results, _ = process_analyze(
        args.infiles,
        args.combined_asl,
        args.cms_file,
        args.workers
    )
    ll_results, _ = process_analyze(
        args.infiles,
        args.ligand_asl,
        args.cms_file,
        args.workers
    )

    # Filter protein-ligand by removing overlaps with both ligand-ligand and protein-protein
    filtered_pl = filter_protein_ligand(pl_results, ll_results, pp_results)
    dicts_to_csv_pandas(filtered_pl, 'protein-ligand', args.outname, total_frames)

    # Write ligand-ligand results
    dicts_to_csv_pandas(ll_results, 'ligand-ligand', args.outname, total_frames)

    duration = datetime.now() - start_time
    print(f"Total runtime: {duration}")

if __name__ == '__main__':
    main()
