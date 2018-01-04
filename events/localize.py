import logging
import multiprocessing
import time
from scipy import ndimage
from lib.constants import *
import numpy as np
import os
import lib
from tqdm import tqdm

logger = logging.getLogger('root')


def localize_events(cluster_matrix, cluster_info, method):
    logger.info("Started event localization on %d events using method %s" % (len(cluster_info), method))
    begin = time.time()

    events = np.empty(len(cluster_info), dtype=dt_event)

    if method == "centroid":
        centroid(cluster_matrix, cluster_info, events)
    elif method == "cnn":
        cnn(cluster_matrix, cluster_info, events)

    time_taken = time.time() - begin

    logger.info(
        "Finished event localization in %d seconds ( %d events/s )" % (time_taken, len(cluster_info) / time_taken))

    return events


def centroid(cluster_matrix, cluster_info, events):
    # Setup pool
    pool = multiprocessing.Pool(lib.config.settings.cores, initializer=lib.init_worker)
    results = list()

    # Progress
    progress_bar = tqdm(total=len(cluster_info), unit="clusters", smoothing=0.1, unit_scale=True)

    # First split clusters in chunks
    # TODO: make this configurable?
    chunk_size = 10000
    if chunk_size > len(cluster_info):
        logger.warn("Cluster chunk size is larger than amount of events")
        chunk_size = len(cluster_info)

    # TODO: Are the same size groups created here?!?
    cm_groups = np.array_split(cluster_matrix, len(cluster_matrix) / chunk_size)
    ci_groups = np.array_split(cluster_info, len(cluster_info) / chunk_size)

    for idx, cm_group in enumerate(cm_groups):
        results.append(pool.apply_async(calculate_centroid, args=([cm_group, ci_groups[idx]])))

    pool.close()

    offset = 0
    for r in results:
        events_chunk = r.get(timeout=100)
        progress_bar.update(len(events_chunk))

        events[offset:offset + len(events_chunk)] = events_chunk
        offset += len(events_chunk)

    progress_bar.close()

    return events


def calculate_centroid(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    # Raise runtime warnings, instead of printing them
    numpy.seterr(all='raise')

    for idx, cluster in enumerate(cluster_matrix):
        # Center of mass of ToT cluster
        try:
            x, y = ndimage.measurements.center_of_mass(cluster[0])
        except FloatingPointError:
            logger.warn("Could not calculate center of mass: empty cluster. Cluster_info: %g" % cluster_info[idx])
            x, y = 0, 0

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # The center_of_mass function considers the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events


def cnn(cluster_matrix, cluster_info, events):
    from keras.models import load_model

    n = len(cluster_info)

    x_test, y_test = np.zeros((n, 2, 4, 4)), np.zeros((n, 2))

    for i in cluster_matrix:
        x_test[i, 0] = cluster_matrix[i, 0, 0:4, 0:4]
        x_test[i, 1] = np.nan_to_num(cluster_matrix[i, 1, 0:4, 0:4])

    package_directory = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(package_directory, 'cnn_models', 'model-tottoa-g4medipix-300si-200kev-set10.h5'))

    # TODO: Make this configurable
    pred = model.predict(x_test, batch_size=(10 ^ 6))

    for idx, p in enumerate(pred):
        events[idx]['chipId'] = cluster_info[idx]['chipId']
        events[idx]['x'] = cluster_info[idx]['x'] + p[0]
        events[idx]['y'] = cluster_info[idx]['y'] + p[1]
        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events
