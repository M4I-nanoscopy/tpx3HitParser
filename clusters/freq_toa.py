import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('root')

def build_freq_toa(cluster_index, hits):
    logger.info("Determine and store ToA frequency matrix")

    # Progress bar
    progress_bar = tqdm(total=len(cluster_index), unit="clusters", smoothing=0.5, unit_scale=True)

    freq_delta_toa = np.zeros((512 * 512, 128), dtype='uint16')

    for cluster_i in cluster_index:
        cluster_i = cluster_i[cluster_i > 0]

        cluster_hits = np.take(hits, cluster_i)

        dtoa = cluster_hits['cToA'] - min(cluster_hits['cToA'])
        vector_idx = cluster_hits['chipId'] * (256 * 256) + cluster_hits['x'] + cluster_hits['y'] * 256

        for idx, i in enumerate(cluster_i):
            freq_delta_toa[vector_idx[idx]][dtoa[idx]] += 1

        progress_bar.update(1)

    progress_bar.close()

    return freq_delta_toa
