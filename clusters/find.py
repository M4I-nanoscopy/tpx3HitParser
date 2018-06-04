from tqdm import tqdm
import lib
import logging
from multiprocessing import Process, JoinableQueue
import time
from lib.constants import *
import numpy as np

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
    results = JoinableQueue()
    inputs = JoinableQueue()

    # Build workers to process the input
    workers = []
    for i in range(lib.config.settings.cores):
        p = Process(target=cluster_worker, args=(inputs, results, lib.config.settings))
        p.start()
        workers.append(p)

    # First split hits in chunks defined by cluster_chunk_size
    if lib.config.settings.cluster_chunk_size > len(hits):
        logger.warning("Cluster chunk size is larger than amount of hits")
        lib.config.settings.cluster_chunk_size = len(hits)

    r = 0
    start = 0
    while start < max_hits:
        end = start + lib.config.settings.cluster_chunk_size

        if end > max_hits:
            end = max_hits

        inputs.put([start, end])

        start = end
        r += 1

    # Build result worker to process results to file
    result_worker = Process(target=cluster_results_worker, args=(inputs, results, progress_bar, lib.config.settings))
    result_worker.start()

    inputs.join()

    time_taken = time.time() - begin

    #logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
    #    clusters, max_hits, time_taken, lib.config.settings.cores, max_hits / time_taken))

def cluster_results_worker(inputs, results, progress_bar, settings):

    while True:
        clusters = results.get()
        progress_bar.update(settings.cluster_chunk_size)

    #
    # clusters = 0
    # for idx in range(0, len(results)):
    #     ci, cm, s = results[idx].get(timeout=1000)
    #
    #     clusters += len(ci)
    #
    #     yield ci, cm, s
    #
    #     # Do not keep previous results object, reduces memory
    #     del results[idx]

def cluster_worker(inputs, results, settings):
    io = lib.io()
    hits = io.read_hits(settings.hits)

    while True:
        hits_range = inputs.get()
        hits_chunk = hits[hits_range[0]:hits_range[1]]

        clusters = find_cluster_matches(settings, hits_chunk, hits_range[0])

        results.put(clusters)


def find_cluster_matches(settings, hits, hits_start):
    # TODO: Move this var to configuration options
    time_size = 50

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
    match_t = (np.absolute(diff_t) < time_size)

    # Look for events from the same chip
    match_c = (diff_c == 0)

    # Combine these condition into one match matrix
    matches = np.logical_and.reduce((match_c, match_x, match_y, match_t))

    # We have to build the cluster matrices already here, and resize later
    clusters = np.zeros((len(hits), 16), dtype='int64')
    # cluster_stats = list()
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
            # Find the indeces and increase with the outside hit index
            cluster_indeces = np.where(select)[0] + hits_start

            # Fill up the cluster with zeros
            clusters[c] = np.append(cluster_indeces, np.zeros((16 - len(cluster_indeces)), dtype='int64'))

            c += 1

        # Build cluster stats if requested
        # if settings.cluster_stats is True:
        #    stats = [len(cluster), np.sum(cluster['ToT'])]
        #    cluster_stats.append(stats)

    # # We made the cluster_info and cluster_matrix too big, resize to actual size
    clusters = np.resize(clusters, c)

    return clusters


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

    # Base cTOA value
    min_ctoa = min(c['cToA'])
    c['cToA'] = c['cToA'] - min_ctoa

    # Base x and y value
    min_x = min(c['x'])
    min_y = min(c['y'])

    # Build cluster_info array
    ci['chipId'] = c[0]['chipId']
    ci['x'] = min_x
    ci['y'] = min_y
    ci['TSPIDR'] = c[0]['TSPIDR']
    ci['cToA'] = min_ctoa

    return ci


class ClusterSizeExceeded(Exception):
    pass

