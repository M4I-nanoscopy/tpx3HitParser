import ctypes
import logging
import multiprocessing
import time
from lib.constants import *
import numpy as np
import scipy.sparse

logger = logging.getLogger('root')

shared_hits = None


def find_clusters(hits, cores):
    logger.info("Started finding clusters")

    # Create memory shared array of the hits. This is much faster then passing the whole array
    # This needs to happen BEFORE building the mp.Pool()
    # TODO: This will NOT work on Windows as global vars are not shared across processes
    global shared_hits
    shared_hits = hits

    pool = multiprocessing.Pool(cores)

    end = 0
    start = 0
    size = len(hits)

    results = list()

    begin = time.time()

    # First split in chunks splits across the number of cores available
    chunk_size = len(hits) / cores + cores
    while end < (size - 1):
        end = end + chunk_size

        if end > size:
            end = size - 1

        results.append(pool.apply_async(find_clusters_chunked, args=(start, end)))
        start = end + 1

    pool.close()

    cluster_info = list()
    cluster_matrix = list()

    for r in results:
        ci, cm = r.get(timeout=999999)

        cluster_info.extend(ci)
        cluster_matrix.extend(cm)

    time_taken = time.time() - begin

    logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
        len(cluster_info), size, time_taken, cores, size / time_taken))

    return cluster_info, cluster_matrix


def find_clusters_chunked(start, end):
    logger.debug("Starting finding clusters of hits %d to %d" % (start, end))

    # Get access to mem shared hits
    global shared_hits
    # Split in chunks of cluster_chunk_size. The find_cluster_matches is memory intensive
    hits_chunk = np.empty((end - start, 8), 'uint16')
    hits_chunk[:] = shared_hits[start:end]

    # Split in groups of cluster_chunk_size
    groups = np.array_split(hits_chunk, len(hits_chunk) / cluster_chunk_size)

    cluster_info = list()
    cluster_matrix = list()

    for group in groups:
        ci, cm = find_cluster_matches(group)
        cluster_info.extend(ci)
        cluster_matrix.extend(cm)

    return cluster_info, cluster_matrix


def find_cluster_matches(hits):
    # TODO: Move this var to configuration options
    timeSize = 50

    x = hits[:, X]
    y = hits[:, Y]
    t = hits[:, cTOA]
    c = hits[:, CHIP]

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
    match_t = (np.absolute(diff_t) < timeSize)

    # Look for events from the same chip
    match_c = (diff_c == 0)

    # Combine these condition into one match matrix
    matches = np.logical_and.reduce((match_c, match_x, match_y, match_t))

    cluster_info = list()
    cluster_matrix = list()

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

        # Only use clean clusters
        if clean_cluster(cluster):
            ci, cm = build_cluster(cluster)
            cluster_info.append(ci)
            cluster_matrix.append(cm)

        # Make sure the events we used are not being used a second time
        matches[select] = False

    return cluster_info, cluster_matrix


# Clean clusters based on their summed ToT and cluster size
def clean_cluster(c):
    sum = np.sum(c[:, TOT])
    size = len(c)

    # Move this var to config options
    if 200 < sum < 400 and 2 < size < 10:
        return True
    else:
        return False


# This builds a cluster from an event list, and the corresponding cluster_info array
def build_cluster(c):
    ci = np.zeros((5), 'uint16')
    cluster = np.zeros((2, n_pixels, n_pixels), 'uint16')

    # Base cTOA value
    min_ctoa = min(c[:, cTOA])
    c[:, cTOA] = c[:, cTOA] - min_ctoa

    # Base x and y value
    min_x = min(c[:, X])
    min_y = min(c[:, Y])

    c[:, X] = c[:, X] - min_x
    c[:, Y] = c[:, Y] - min_y

    rows = c[:, X]
    cols = c[:, Y]
    ToT = c[:, TOT]
    ToA = c[:, cTOA]

    try:
        cluster[0, :, :] = scipy.sparse.coo_matrix((ToT, (rows, cols)), shape=(n_pixels, n_pixels)).todense()
        cluster[1, :, :] = scipy.sparse.coo_matrix((ToA, (rows, cols)), shape=(n_pixels, n_pixels)).todense()
    except ValueError:
        logger.warn("Cluster exceeded max cluster size defined by n_pixels (%i)" % (n_pixels))

    ci[CI_CHIP] = c[0][CHIP]
    ci[CI_X] = min_x
    ci[CI_Y] = min_y
    ci[CI_SPIDR_TIME] = c[0][SPIDR_TIME]
    ci[CI_cTOA] = min_ctoa

    return ci, cluster
