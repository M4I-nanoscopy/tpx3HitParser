from tqdm import tqdm
import lib
import logging
import multiprocessing
import time
from lib.constants import *
import numpy as np
import scipy.sparse

logger = logging.getLogger('root')


def find_clusters(hits, cores):
    logger.info("Started finding clusters")

    pool = multiprocessing.Pool(cores)
    results = list()
    begin = time.time()

    # First split in chunks splits across the number of cores available
    groups = np.array_split(hits, len(hits) / lib.config.settings.cluster_chunk_size)

    for group in groups:
        # TODO: By passing the the chunk of hits (group) as argument, a large amount of pickle data is passed over
        # the process bus. This could be improved maybe with better shared mem usage
        results.append(pool.apply_async(find_cluster_matches, args=([group])))

    pool.close()

    cluster_info = list()
    cluster_matrix = list()

    progress_bar = tqdm(total=len(hits), unit="hits", smoothing=0.1, unit_scale=True)
    filtered = 0

    for r in results:
        ci, cm, f = r.get(timeout=100)
        progress_bar.update(len(groups[0]))

        cluster_info.extend(ci)
        cluster_matrix.extend(cm)
        filtered += f

    time_taken = time.time() - begin

    logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
        len(cluster_info), len(hits), time_taken, cores, len(hits) / time_taken))

    filtered_percentage = float(filtered / float(len(cluster_info) + filtered)) * 100
    logger.info("Filtered %d clusters (%d percent)" % (filtered, filtered_percentage))

    return cluster_info, cluster_matrix


def find_cluster_matches(hits):
    # TODO: Move this var to configuration options
    time_size = 50

    x = hits['x']
    y = hits['y']
    t = hits['cToA']
    c = hits['chipId']

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

    cluster_info = list()
    cluster_matrix = list()
    filtered = 0

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
        elif len(cluster) > 1:
            filtered += 1

        # Make sure the events we used are not being used a second time
        matches[select] = False

    return cluster_info, cluster_matrix, filtered


# Clean clusters based on their summed ToT and cluster size
def clean_cluster(c):
    summed = np.sum(c['ToT'])
    size = len(c)

    settings = lib.config.settings

    if settings.cluster_min_sum_tot < summed < settings.cluster_max_sum_tot \
            and settings.cluster_min_size < size < settings.cluster_max_size:
        return True
    else:
        return False


# This builds a cluster from an event list, and the corresponding cluster_info array
def build_cluster(c):
    m_size = lib.config.settings.cluster_matrix_size
    ci = np.zeros(1, dtype=dt_ci)
    cluster = np.zeros((2, m_size, m_size), 'uint8')

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
        cluster[0, :, :] = scipy.sparse.coo_matrix((tot, (rows, cols)), shape=(m_size, m_size)).todense()
        cluster[1, :, :] = scipy.sparse.coo_matrix((toa, (rows, cols)), shape=(m_size, m_size)).todense()
    except ValueError:
        logger.warn("Cluster exceeded max cluster size defined by cluster_matrix_size (%i)" % m_size)

    ci['chipId'] = c[0]['chipId']
    ci['x'] = min_x
    ci['y'] = min_y
    ci['TSPIDR'] = c[0]['TSPIDR']
    ci['cToA'] = min_ctoa

    return ci, cluster
