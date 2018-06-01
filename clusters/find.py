from scipy.spatial.ckdtree import cKDTree
from tqdm import tqdm
import lib
import logging
from multiprocessing import Process, JoinableQueue, Queue
import time

from lib import io
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

        # Reads hits chunk wise
        #hits_chunk = hits[start:end]

        inputs.put([start, end])

        start = end
        r += 1

    # Build result worker to process results to file
    result_worker = Process(target=cluster_results_worker, args=(inputs, results, progress_bar))
    result_worker.start()

    workers[0].join()


    time_taken = time.time() - begin

    #logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
    #    clusters, max_hits, time_taken, lib.config.settings.cores, max_hits / time_taken))

def cluster_results_worker(inputs, results, progress_bar):

    while True:
        hits = results.get()
        progress_bar.update(hits)

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
        # hits_chunk = inputs.get()
        find_cluster_matches(settings, hits_chunk)

        results.put(len(hits_chunk))


def find_cluster_matches(settings, hits):
    # TODO: Move this var to configuration options
    time_size = 50

    m = np.column_stack((hits['x'], hits['y'], hits['cToA'] / 50, hits['chipId']))

    t = cKDTree(m, 30)
    clusters = t.query_ball_tree(t, 2)

    # We have to build the cluster matrices already here, and resize later
    cluster_info = np.zeros(len(hits), dtype=dt_ci)
    cluster_stats = list()
    c = 0

    # Loop over all columns of matches, and handle event/cluster per column
    for cluster_select in clusters:
        cluster = hits[cluster_select]

        # Only use clean clusters
        ci = build_cluster(cluster, settings)
        cluster_info[c] = ci
        c += 1

        # Build cluster stats if requested
        if settings.cluster_stats is True:
            stats = [len(cluster), np.sum(cluster['ToT'])]
            cluster_stats.append(stats)

    # # We made the cluster_info and cluster_matrix too big, resize to actual size
    cluster_info = np.resize(cluster_info, c)

    return cluster_info, cluster_stats


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

