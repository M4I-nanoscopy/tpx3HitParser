import _tkinter
from matplotlib import patches
from tqdm import tqdm
import lib
import logging
import multiprocessing
import time
from lib.constants import *
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

logger = logging.getLogger('root')


def find_clusters(hits):
    logger.info("Started finding clusters")

    pool = multiprocessing.Pool(lib.config.settings.cores, initializer=lib.init_worker, maxtasksperchild=100)
    results = {}
    begin = time.time()

    # First split hits in chunks defined by cluster_chunk_size
    if lib.config.settings.cluster_chunk_size > len(hits):
        logger.warning("Cluster chunk size is larger than amount of hits")
        lib.config.settings.cluster_chunk_size = len(hits)

    r = 0
    start = 0
    while start < len(hits):
        end = start + lib.config.settings.cluster_chunk_size

        if end > len(hits):
            end = len(hits)

        # Reads hits chunk wise
        hits_chunk = hits[start:end]

        # The `lib.config.settings` is passed here as Windows will not pass it as global var
        results[r] = pool.apply_async(find_cluster_matches, args=([lib.config.settings, hits_chunk]))

        start = end
        r += 1

    pool.close()

    progress_bar = tqdm(total=len(hits), unit="hits", smoothing=0.1, unit_scale=True)

    clusters = 0
    for idx in range(0, len(results)):
        ci, cm, s = results[idx].get(timeout=1000)
        progress_bar.update(lib.config.settings.cluster_chunk_size)

        clusters += len(ci)

        yield ci, cm, s

        # Do not keep previous results object, reduces memory
        del results[idx]

    time_taken = time.time() - begin

    logger.info("Finished finding %d clusters from %d hits in %d seconds on %d cores ( %d hits / second ) " % (
        clusters, len(hits), time_taken, lib.config.settings.cores, len(hits) / time_taken))


def find_cluster_matches(settings, hits):
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
    cluster_info = np.zeros(len(hits), dtype=dt_ci)
    cluster_matrix = np.zeros((len(hits), 2, settings.cluster_matrix_size, settings.cluster_matrix_size), 'uint8')
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

        # Only use clean clusters
        if clean_cluster(cluster, settings):
            try:
                ci, cm = build_cluster(cluster, settings)
                cluster_info[c] = ci
                cluster_matrix[c] = cm
                c += 1
            except ClusterSizeExceeded:
                logger.warning("Cluster exceeded max cluster size "
                               "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
                pass

        # Build cluster stats if requested
        if settings.cluster_stats is True and len(cluster) > 0:
            stats = [len(cluster), np.sum(cluster['ToT'])]
            cluster_stats.append(stats)

        # Make sure the events we used are not being used a second time
        matches[select] = False

    # We made the cluster_info and cluster_matrix too big, resize to actual size
    cluster_info = np.resize(cluster_info, c)
    cluster_matrix = np.resize(cluster_matrix, (c, 2, settings.cluster_matrix_size, settings.cluster_matrix_size))

    return cluster_info, cluster_matrix, cluster_stats


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

    return ci, cluster


class ClusterSizeExceeded(Exception):
    pass


def print_cluster_stats(cluster_info, cluster_stats):
    cluster_stats = np.array(cluster_stats)

    removed = len(cluster_stats) - len(cluster_info)
    removed_percentage = float(removed / float(len(cluster_info) + removed)) * 100

    logger.info("Removed %d clusters and single hits (%d percent)" % (removed, removed_percentage))

    # Figure
    try:
        fig, ax = plt.subplots()
    except _tkinter.TclError as e:
        logger.error('Could not display cluster_stats plot. Error message was: %s' % str(e))
        return

    # Make 2d hist
    cmap = plt.get_cmap('viridis')
    cmap.set_under('w', 1)
    bins = [np.arange(0, 700, 50), np.arange(0, 16, 1)]
    plt.hist2d(cluster_stats[:, 1], cluster_stats[:, 0], cmap=cmap, vmin=1, range=((0, 700), (0, 16)), bins=bins)

    # Add box showing filter values
    settings = lib.config.settings
    ax.add_patch(
        patches.Rectangle(
            (settings.cluster_min_sum_tot, settings.cluster_min_size),  # (x,y)
            settings.cluster_max_sum_tot - settings.cluster_min_sum_tot,  # width
            settings.cluster_max_size - settings.cluster_min_size,  # height
            fill=False, edgecolor='red', linewidth=2
        )
    )

    ax.set_xticks(bins[0])
    ax.set_yticks(bins[1])
    ax.set_ylim(1)
    plt.tick_params(colors='black', )
    plt.grid(b=True, which='both')
    plt.ylabel('Cluster Size')
    plt.xlabel('Cluster ToT sum')
    plt.colorbar()

    # plt.hist(cluster_stats[:, 1], range=(0, 700), bins=699)

    plt.show()
