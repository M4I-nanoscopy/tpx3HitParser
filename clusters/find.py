import signal
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
    # Recast to signed integers, as we need to subtract
    # TODO: This casting causes a lot of extra memory to be used, can we do this better?
    x = hits['x'].astype('int16')
    y = hits['y'].astype('int16')
    t = hits['cToA'].astype('int32')
    c = hits['chipId'].astype('int8')

    # Calculate for all events the difference in x, y, cTOA and chip with all other event
    # This is a memory intensive step! We're creating 4 times a cluster_chunk_size * cluster_chunk_size sized matrix
    diff_x = x.reshape(-1, 1) - x
    diff_y = y.reshape(-1, 1) - y
    diff_t = t.reshape(-1, 1) - t
    diff_c = c.reshape(-1, 1) - c

    # Look for events that are right next to our current pixel
    match_x = np.logical_and(diff_x < 2, diff_x > -2)
    match_y = np.logical_and(diff_y < 2, diff_y > -2)

    # Look for events which are close in ToA
    match_t = (np.absolute(diff_t) < settings.cluster_time_window)

    # Look for events from the same chip
    match_c = (diff_c == 0)

    # Combine these condition into one match matrix
    matches = np.logical_and.reduce((match_c, match_x, match_y, match_t))

    # We have to build the cluster matrices already here, and resize later
    index_chunk = np.zeros((len(hits), 16), dtype='int64')
    ci_chunk = np.zeros(len(hits), dtype=dt_ci)
    cm_chunk = np.zeros((len(hits), 2, settings.cluster_matrix_size, settings.cluster_matrix_size), dt_clusters)
    cluster_stats = list()
    c = 0

    # Loop over all columns of matches, and handle event/cluster per column
    for m in range(0, matches.shape[0]):
        select = matches[:, m]

        # Select all the matches of the events of this initial event
        selected = matches[select]

        prev_len = -1
        # Find all events that belong to this cluster, but are not directly connected to the event we started with
        while prev_len != len(selected):
            prev_len = len(selected)
            select = np.any(selected.transpose(), axis=1)
            selected = matches[select]

        cluster = hits[select]

        # Make sure the events we used are not being used a second time
        matches[select] = False

        # Only use clean clusters
        if len(cluster) > 0 and clean_cluster(cluster, settings):
            try:
                ci, cm = build_cluster(cluster, settings)
                ci_chunk[c] = ci
                cm_chunk[c] = cm
            except lib.ClusterSizeExceeded:
                logger.warning("Cluster exceeded max cluster size "
                               "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
                pass

            if settings.store_cluster_indices:
                # Find the indices and increase with the outside hit index
                cluster_indices = np.where(select)[0] + hits_start

                # Fill up the cluster with zeros
                index_chunk[c] = np.append(cluster_indices, np.zeros((16 - len(cluster_indices))))

            c += 1

        # Build cluster stats if requested
        if settings.cluster_stats is True and len(cluster) > 0:
            stats = [len(cluster), np.sum(cluster['ToT'])]
            cluster_stats.append(stats)

    # We made the cluster_info and cluster_matrix too big, resize to actual size
    index_chunk = np.resize(index_chunk, (c, 16))
    ci_chunk = np.resize(ci_chunk, c)
    cm_chunk = np.resize(cm_chunk, (c, 2, settings.cluster_matrix_size, settings.cluster_matrix_size))

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

    rows = c['y']
    cols = c['x']
    tot = c['ToT']
    toa = c['cToA']

    try:
        # TODO: We're throwing away NaN information here of pixels that have not been hit, but this function is fast!
        cluster[0, :, :] = scipy.sparse.coo_matrix((tot, (rows, cols)), shape=(m_size, m_size)).todense()
        cluster[1, :, :] = scipy.sparse.coo_matrix((toa, (rows, cols)), shape=(m_size, m_size)).todense()
    except ValueError:
        raise lib.ClusterSizeExceeded

    # Build cluster_info array
    ci['chipId'] = c[0]['chipId']
    ci['x'] = min_x
    ci['y'] = min_y
    ci['TSPIDR'] = c[0]['TSPIDR']
    ci['cToA'] = min_ctoa
    ci['sumToT'] = np.sum(c['ToT'])

    return ci, cluster


