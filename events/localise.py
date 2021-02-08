import logging
import multiprocessing
import time
import random
from scipy import ndimage

import tpx3format
from lib.constants import *
import numpy as np
import os
import lib
from tqdm import tqdm

# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0

logger = logging.getLogger('root')


def localise_events(cluster_matrix, cluster_info, method):
    logger.info("Started event localization on %d events using method %s" % (len(cluster_info), method))
    begin = time.time()

    events = np.empty(len(cluster_info), dtype=dt_event)

    if method == "centroid":
        events = split_calculation(cluster_matrix, cluster_info, events, calculate_centroid)
    elif method == "random":
        events = split_calculation(cluster_matrix, cluster_info, events, calculate_random)
    elif method == "highest_toa":
        events = split_calculation(cluster_matrix, cluster_info, events, calculate_toa)
    elif method == "highest_tot":
        events = split_calculation(cluster_matrix, cluster_info, events, calculate_tot)
    elif method == "cnn":
        events = cnn(cluster_matrix, cluster_info, events, lib.config.settings.event_cnn_tot_only)
    else:
        raise Exception("Chosen localisation algorithm ('%s') does not exist" % method)

    time_taken = time.time() - begin

    logger.info(
        "Finished event localization in %d seconds ( %d events/s )" % (time_taken, len(cluster_info) / time_taken))

    return events


def split_calculation(cluster_matrix, cluster_info, events, method):
    # Setup pool
    pool = multiprocessing.Pool(lib.config.settings.cores, initializer=lib.init_worker, maxtasksperchild=1000)
    results = {}

    # Progress bar
    progress_bar = tqdm(total=len(cluster_info), unit="clusters", smoothing=0.1, unit_scale=True)

    # First split clusters in chunks
    chunk_size = lib.config.settings.event_chunk_size
    if chunk_size > len(cluster_info):
        logger.warning("Cluster chunk size is larger than amount of events")
        chunk_size = len(cluster_info)

    start = 0
    r = 0
    while start < len(cluster_info):
        end = start + chunk_size

        if end > len(cluster_info):
            end = len(cluster_info)

        # This reads the clusters chunk wise.
        cm_chunk = cluster_matrix[start:end]
        ci_chunk = cluster_info[start:end]

        results[r] = pool.apply_async(method, args=([cm_chunk, ci_chunk]))
        start = end
        r += 1

    pool.close()

    offset = 0
    for idx in range(0, len(results)):
        events_chunk = results[idx].get(timeout=100)
        progress_bar.update(len(events_chunk))

        events[offset:offset + len(events_chunk)] = events_chunk
        offset += len(events_chunk)

        del results[idx]

    progress_bar.close()

    return events


def calculate_centroid(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    # Raise runtime warnings, instead of printing them
    numpy.seterr(all='raise')

    for idx, cluster in enumerate(cluster_matrix):
        # Center of mass of ToT cluster
        try:
            y, x = ndimage.measurements.center_of_mass(cluster[0])
        except FloatingPointError:
            logger.warning("Could not calculate center of mass: empty cluster. Cluster_info: %s" % cluster_info[idx])
            y, x = 0, 0

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # The center_of_mass function considers the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']
        events[idx]['sumToT'] = cluster_info[idx]['sumToT']

    return events


def calculate_random(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    for idx, cluster in enumerate(cluster_matrix):
        nzy, nzx = np.nonzero(cluster[0])

        if len(nzy) == 0:
            logger.warning("Could not find random pixel: empty cluster?. Cluster_idx: %s" % idx)
            continue
        elif len(nzy) == 1:
            y, x = nzy[0], nzx[0]
        else:
            i = np.random.randint(0, len(nzy) - 1)
            y, x = nzy[i], nzx[i]

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # We considered the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5 + random.uniform(-0.5, 0.5)
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5 + random.uniform(-0.5, 0.5)

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']
        events[idx]['sumToT'] = cluster_info[idx]['sumToT']

    return events


def calculate_toa(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    for idx, cluster in enumerate(cluster_matrix):
        if np.max(cluster[1]) == 0:
            logger.debug("Could not calculate highest_toa: empty ToA cluster Picking random ToT pixel. Cluster_info: %s" % cluster_info[idx])

            if np.max(cluster[0]) == 0:
                logger.warning("ToT and ToA cluster empty. Giving up for this cluster. Cluster_info: %s" %cluster_info[idx])
                continue

            nzy, nzx = np.nonzero(cluster[0])
            i = np.random.randint(0, len(nzy) - 1)
            y, x = nzy[i], nzx[i]
        else:
            maxes = np.argwhere(cluster[1] == np.max(cluster[1]))

            if len(maxes) > 1:
                i = np.random.randint(0, len(maxes) - 1)
                y, x = maxes[i][0], maxes[i][1]
            else:
                y, x = maxes[0][0], maxes[0][1]

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # We considered the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events


def calculate_tot(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    for idx, cluster in enumerate(cluster_matrix):
        i = np.argmax(cluster[0])
        y, x = np.unravel_index(i, cluster[0].shape)

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # We considered the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']
        events[idx]['sumToT'] = cluster_info[idx]['sumToT']

    return events


def cnn(cluster_matrix, cluster_info, events, tot_only):
    # Do keras and tensorflow imports here, as importing earlier may raise errors unnecessary
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # Hide some of the TensorFlow debug information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set amount of cores to use for TensorFlow when using CPU only
    tf.config.threading.set_intra_op_parallelism_threads(lib.config.settings.cores)

    # Load model
    model_path = lib.config.settings.event_cnn_model

    if not os.path.exists(model_path):
        raise lib.UserConfigException('CNN model %s does not exist.' % model_path)

    model = load_model(model_path)

    # Delete ToA matrices, required for ToT only CNN
    if tot_only:
        cluster_matrix = np.delete(cluster_matrix, 1, 1)

    # Check model shape and input shape
    if cluster_matrix.shape[1:4] != model.layers[0].input_shape[1:4]:
        logger.error(
            'Cluster matrix shape %s does not match model shape %s. Change cluster_matrix_size or use --event_cnn_tot_only?' % (
                cluster_matrix.shape, model.layers[0].input_shape))
        raise Exception

    # Run CNN prediction
    predictions = model.predict(cluster_matrix, batch_size=lib.config.settings.event_chunk_size, verbose=1)

    # Copy all events from cluster_info as base
    # TODO: This loads whole cluster_info matrix at once and may cause memory issues
    events = cluster_info[()].astype(dt_event)

    # Add prediction offset from cluster origin
    events['x'] = events['x'] + predictions[:, 1]
    events['y'] = events['y'] + predictions[:, 0]

    shape = tpx3format.calculate_image_shape()

    # Check for events outside matrix shape, and delete those
    ind_del = (events['x'] > shape) | (events['y'] > shape)
    indices = np.arange(len(events))
    events = np.delete(events, indices[ind_del], axis=0)
    deleted = np.count_nonzero(ind_del)

    if deleted > 0:
        logger.warning('Removed %d events found outside image matrix shape (%d).' % (deleted, shape))

    return events
