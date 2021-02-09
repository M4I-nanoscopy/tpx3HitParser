import lib
import logging

from clusters.clfind_rust import clfind
from lib.constants import *
import numpy as np
import scipy.sparse

logger = logging.getLogger('root')


def find_clusters(settings, hits):
    # Outsource the main cluster finding routine to a Rust compiled library
    hits_stacked = np.stack((hits['x'], hits['y'], hits['cToA']), axis=-1).astype('int64')
    labels = clfind(hits_stacked)

    # This takes the cluster labels, and take their hits, and converts it into a list of clusters with their hits
    idx = labels.argsort()
    ls = labels[idx]
    split = 1 + np.where(ls[1:] != ls[:-1])[0]
    clusters = np.split(hits[idx], split)

    # Build the maximum size clusters we expect, before filtering
    cm_chunk = np.zeros((len(clusters), 2, settings.cluster_matrix_size, settings.cluster_matrix_size), dt_clusters)
    ci_chunk = np.zeros(len(clusters), dtype=dt_ci)

    # Loop over all matches
    c = 0
    for cluster in clusters:
        # Only use clean clusters
        if not clean_cluster(cluster, settings):
            continue

        try:
            ci, cm = build_cluster(cluster, settings)
            cm_chunk[c] = cm
            ci_chunk[c] = ci
            c += 1
        except lib.ClusterSizeExceeded:
            logger.warning("Cluster exceeded max cluster size "
                           "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
            pass

    ci_chunk = np.resize(ci_chunk, c)
    cm_chunk = np.resize(cm_chunk, (c, 2, settings.cluster_matrix_size, settings.cluster_matrix_size))

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

    # Base x and y value
    min_x = min(c['x'])
    min_y = min(c['y'])

    rows = c['y'] - min_y
    cols = c['x'] - min_x
    dtoa = c['cToA'] - min_ctoa

    try:
        # TODO: We're throwing away the information which pixels have not been hit, but this function is fast!
        cluster[0, :, :] = scipy.sparse.coo_matrix((c['ToT'], (rows, cols)), shape=(m_size, m_size)).todense()
        cluster[1, :, :] = scipy.sparse.coo_matrix((dtoa, (rows, cols)), shape=(m_size, m_size)).todense()
    except ValueError:
        raise lib.ClusterSizeExceeded

    # Build cluster_info array
    ci['chipId'] = c[0]['chipId']
    ci['x'] = min_x
    ci['y'] = min_y
    ci['TSPIDR'] = c[0]['TSPIDR']
    ci['cToA'] = min_ctoa

    return ci, cluster


