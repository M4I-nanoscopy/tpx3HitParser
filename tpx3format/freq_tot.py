import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
import lib

logger = logging.getLogger('root')


def build_freq_tot(hits, cores):
    logger.info("Determine and store ToT frequency matrix")

    # Setup pool
    pool = multiprocessing.Pool(cores)
    results = {}

    # Progress bar
    progress_bar = tqdm(total=len(hits), unit="hits", smoothing=0.1, unit_scale=True)

    # First split hits in chunks
    chunk_size = int(len(hits) / cores / 5)
    if chunk_size > len(hits):
        logger.warning("Hits chunk size is larger than amount of hits")
        chunk_size = len(hits)

    start = 0
    r = 0
    while start < len(hits):
        end = start + chunk_size

        if end > len(hits):
            end = len(hits)

        # This reads the clusters chunk wise.
        hits_chunk = hits[start:end]

        results[r] = pool.apply_async(calc_freq_tot, args=[hits_chunk])
        start = end
        r += 1

    pool.close()

    freq_tot = np.zeros((512 * 512, 1024), dtype='uint32')

    for idx in range(0, len(results)):
        freq_tot_chunk = results[idx].get(timeout=1000)

        progress_bar.update(chunk_size)

        freq_tot += freq_tot_chunk

        del results[idx]

    progress_bar.close()

    return freq_tot


def calc_freq_tot(hits):
    vector_idx = hits['chipId'] * (256 * 256) + hits['x'] + hits['y'] * 256

    freq_tot = np.zeros((512*512, 1024), dtype='uint16')

    for idx, hit in enumerate(hits):
        freq_tot[vector_idx[idx]][hit['ToT']] += 1

    return freq_tot
