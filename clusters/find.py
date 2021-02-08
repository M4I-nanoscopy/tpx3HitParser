import lib
import logging

from clusters.clfind_rust import clfind
from lib.constants import *
import numpy as np
import scipy.sparse

logger = logging.getLogger('root')


def find_clusters(settings, hits):
    hits2 = np.stack((hits['x'], hits['y'], hits['cToA']), axis=-1).astype('int64')
    labels = clfind(hits2)

    cm_chunk = np.zeros((len(hits), 2, settings.cluster_matrix_size, settings.cluster_matrix_size), dt_clusters)
    ci_chunk = np.zeros(len(hits), dtype=dt_ci)

    # Loop over all matches
    c = 0
    for label in range(np.max(labels)):
        hit_idx = np.where(labels == label)

        if len(hit_idx[0]) > 2:
            cluster = np.take(hits, hit_idx[0])
        else:
            continue

        # Only use clean clusters
        if clean_cluster(cluster, settings):
            try:
                ci, cm = build_cluster(cluster, settings)
                cm_chunk[c] = cm
                ci_chunk[c] = ci
                c += 1
            except lib.ClusterSizeExceeded:
                logger.warning("Cluster exceeded max cluster size "
                               "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
                pass

    #ci_chunk = np.resize(ci_chunk, c)
    #cm_chunk = np.resize(cm_chunk, (c, 2, settings.cluster_matrix_size, settings.cluster_matrix_size))

    return cm_chunk, ci_chunk


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

    return ci, cluster


