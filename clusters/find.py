import signal

from sklearn.cluster import DBSCAN
from tqdm import tqdm
import lib
import logging
from multiprocessing import Process, Queue
import time
from lib.constants import *
import numpy as np
import scipy.sparse

logger = logging.getLogger('root')


def find_clusters(hits):
    logger.info("Started finding clusters")
    begin = time.time()

    # Respect max_hits setting also here
    if 0 < lib.config.settings.max_hits < len(hits):
        max_hits = lib.config.settings.max_hits
    else:
        max_hits = len(hits)

    progress_bar = tqdm(total=max_hits, unit="hits", smoothing=0.1, unit_scale=True)

    # Build Queues to handle the input data and the resulting clusters
    results = Queue()
    inputs = Queue()

    # Build workers to process the input
    workers = []
    for i in range(lib.config.settings.cores):
        p = Process(target=cluster_worker, args=(inputs, results, lib.config.settings))
        p.daemon = True
        p.start()
        workers.append(p)

    # First split hits in chunks defined by cluster_chunk_size
    if lib.config.settings.cluster_chunk_size > len(hits):
        logger.warning("Cluster chunk size is larger than amount of hits")
        lib.config.settings.cluster_chunk_size = len(hits)

    start = 0
    chunks = 0
    while start < max_hits:
        end = start + lib.config.settings.cluster_chunk_size

        if end > max_hits:
            end = max_hits

        # Build up queue of input
        inputs.put((hits[start:end], start))

        # Increase chunk counter
        chunks += 1

        start = end

    total_clusters = 0
    try:
        # Wait for result worker to finish
        while chunks > 0:
            cm_chunk, ci_chunk, index_chunk, cluster_stats = results.get()

            # Update progress bar
            progress_bar.update(lib.config.settings.cluster_chunk_size)

            # Decrease chunk counter
            chunks -= 1

            total_clusters += len(cm_chunk)

            # Return to be written to file
            yield cm_chunk, ci_chunk, index_chunk, cluster_stats
    except KeyboardInterrupt:
        # The finally block is executed always, but the KeyboardInterrupt needs to be reraised to be handled by parent
        raise KeyboardInterrupt
    finally:
        # Terminate workers and progress bar
        for worker in workers:
            worker.terminate()
            worker.join()
        progress_bar.close()

    time_taken = time.time() - begin

    logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
        total_clusters, max_hits, time_taken, lib.config.settings.cores, max_hits / time_taken))


def cluster_worker(inputs, results, settings):
    # Ignore the interrupt signal. Let parent handle that.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        hits, start_idx = inputs.get()

        # Process to clusters
        cm_chunk, ci_chunk, index_chunk, cluster_stats = find_cluster_matches(settings, hits, start_idx)

        # Put results on result queue
        results.put((cm_chunk, ci_chunk, index_chunk, cluster_stats))


def find_cluster_matches(settings, hits, hits_start):
    # TODO: Move this var to configuration options
    time_size = 50

    # Find clusters using DBSCAN
    m = np.stack((hits['x'], hits['y'], (hits['cToA'] / time_size).astype(int)), axis=-1)
    cluster_labels = DBSCAN(eps=1, min_samples=1, metric='euclidean').fit_predict(m)
    n_clusters = np.max(cluster_labels)

    # Build the cluster matrices to hold the outcome
    index_chunk = np.zeros((n_clusters, 16), dtype='int64')
    ci_chunk = np.zeros(n_clusters, dtype=dt_ci)
    cm_chunk = np.zeros((n_clusters, 2, settings.cluster_matrix_size, settings.cluster_matrix_size), dt_clusters)
    cluster_stats = list()
    c = 0

    # Which clusters were used
    used = list()

    # Loop over all columns of matches, and handle event/cluster per column
    for m in cluster_labels:
        if m in used:
            continue

        used.append(m)
        cluster = np.take(hits, np.where(cluster_labels == m))[0]

        # Only use clean clusters
        if clean_cluster(cluster, settings):
            try:
                ci, cm = build_cluster(cluster, settings)
                ci_chunk[c] = ci
                cm_chunk[c] = cm
            except ClusterSizeExceeded:
                logger.warning("Cluster exceeded max cluster size "
                               "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
                pass

            if settings.store_cluster_indices:
                # Find the indices and increase with the outside hit index
                cluster_indices = m

                # Fill up the cluster with zeros
                index_chunk[c] = np.append(cluster_indices, np.zeros((16 - len(cluster_indices))))

            c += 1

        # Build cluster stats if requested
        if settings.cluster_stats:
            stats = [len(cluster), np.sum(cluster['ToT'])]
            cluster_stats.append(stats)

    return cm_chunk, ci_chunk, index_chunk, cluster_stats


# Clean clusters based on their summed ToT and cluster size
def clean_cluster(c, settings):
    summed = np.sum(c['ToT'])
    size = len(c)

    if settings.cluster_min_sum_tot < summed < settings.cluster_max_sum_tot \
            and settings.cluster_min_size < size < settings.cluster_max_size:
        return True
    else:
        return False


# This builds a cluster from an event list, and the corresponding cluster_info array
def build_cluster(c, settings):
    m_size = settings.cluster_matrix_size
    ci = np.zeros(1, dtype=dt_ci)
    cluster = np.zeros((2, m_size, m_size), dt_clusters)

    # Base cTOA value
    min_ctoa = min(c['cToA'])
    c['cToA'] = c['cToA'] - min_ctoa

    # Base x and y value
    min_x = min(c['x'])
    min_y = min(c['y'])

    c['x'] = c['x'] - min_x
    c['y'] = c['y'] - min_y

    rows = c['x']
    cols = c['y']
    tot = c['ToT']
    toa = c['cToA']

    try:
        # TODO: We're throwing away NaN information here of pixels that have not been hit, but this function is fast!
        cluster[0, :, :] = scipy.sparse.coo_matrix((tot, (rows, cols)), shape=(m_size, m_size)).todense()
        cluster[1, :, :] = scipy.sparse.coo_matrix((toa, (rows, cols)), shape=(m_size, m_size)).todense()
    except ValueError:
        raise ClusterSizeExceeded

    # Build cluster_info array
    ci['chipId'] = c[0]['chipId']
    ci['x'] = min_x
    ci['y'] = min_y
    ci['TSPIDR'] = c[0]['TSPIDR']
    ci['cToA'] = min_ctoa
    ci['sumToT'] = np.sum(c['ToT'])

    return ci, cluster


class ClusterSizeExceeded(Exception):
    pass
