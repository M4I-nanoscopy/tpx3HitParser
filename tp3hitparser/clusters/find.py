import logging

import numpy as np
import scipy.sparse

from tp3hitparser import lib
from tp3hitparser.clusters.clfind import clfind
from tp3hitparser.lib.constants import *

logger = logging.getLogger('root')

def find_clusters(settings, hits):
    # Outsource the main cluster finding routine to a Rust compiled library
    hits_stacked = np.stack((hits['x'], hits['y'], hits['ToA']), axis=-1).astype('int64')
    labels = clfind(hits_stacked, settings.cluster_time_window)

    # This takes the cluster labels, and take their hits, and converts it into a list of clusters with their hits
    idx = labels.argsort()
    ls = labels[idx]
    split = 1 + np.where(ls[1:] != ls[:-1])[0]
    clusters = np.split(hits[idx], split)

    # Build the maximum size clusters we expect, before filtering
    cm_chunk = np.zeros((len(clusters), 2, settings.cluster_matrix_size, settings.cluster_matrix_size), dt_clusters)
    ci_chunk = np.zeros(len(clusters), dtype=cluster_info_datatype(settings.cluster_stats))
    # Loop over all clusters
    i = 0
    for cluster in clusters:
        # Only use filtered clusters
        if not clean_cluster(cluster, settings):
            continue

        try:
            build_cluster(cluster, settings, i, cm_chunk, ci_chunk)
            i += 1
        except lib.ClusterSizeExceeded:
            logger.warning("Cluster exceeded max cluster size "
                           "defined by cluster_matrix_size (%i)" % settings.cluster_matrix_size)
            pass

    ci_chunk = np.resize(ci_chunk, i)
    cm_chunk = np.resize(cm_chunk, (i, 2, settings.cluster_matrix_size, settings.cluster_matrix_size))

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
def build_cluster(c, settings, i, cm, ci):
    m_size = settings.cluster_matrix_size

    # Base cTOA value
    min_ctoa = min(c['ToA'])

    # Base x and y value
    min_x = min(c['x'])
    min_y = min(c['y'])

    rows = c['y'] - min_y
    cols = c['x'] - min_x
    dtoa = c['ToA'] - min_ctoa

    try:
        # TODO: We're throwing away the information which pixels have not been hit, but this function is fast!
        cm[i, 0, :, :] = scipy.sparse.coo_matrix((c['ToT'], (rows, cols)), shape=(m_size, m_size)).todense()
        cm[i, 1, :, :] = scipy.sparse.coo_matrix((dtoa, (rows, cols)), shape=(m_size, m_size)).todense()
    except ValueError:
        raise lib.ClusterSizeExceeded

    # Build cluster_info array
    ci[i]['chipId'] = c[0]['chipId']
    ci[i]['x'] = min_x
    ci[i]['y'] = min_y
    ci[i]['ToA'] = min_ctoa

    if settings.cluster_stats:
        ci[i]['sumToT'] = np.sum(c['ToT'])
        ci[i]['nHits'] = len(c)


def cluster_info_datatype(cluster_stats):
    dt = dt_ci_base

    if cluster_stats:
        dt = dt + dt_ci_extended

    return numpy.dtype(dt)
