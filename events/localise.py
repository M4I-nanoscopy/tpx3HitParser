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
            x, y = ndimage.measurements.center_of_mass(cluster[0])
        except FloatingPointError:
            logger.warning("Could not calculate center of mass: empty cluster. Cluster_info: %s" % cluster_info[idx])
            x, y = 0, 0

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # The center_of_mass function considers the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']
        events[idx]['sumToT'] = np.sum(cluster[0])

    return events


def calculate_random(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    for idx, cluster in enumerate(cluster_matrix):
        non_zeros = np.transpose(np.nonzero(cluster[0]))
        c = random.sample(non_zeros, 1)
        x = c[0][0]
        y = c[0][1]

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # The center_of_mass function considers the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5 + random.uniform(-0.5, 0.5)
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5 + random.uniform(-0.5, 0.5)

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events


def calculate_toa(cluster_matrix, cluster_info):
    events = np.empty(len(cluster_info), dtype=dt_event)

    for idx, cluster in enumerate(cluster_matrix):
        i = np.argmax(cluster[1])
        x,y = np.unravel_index(i,(10,10))

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # The center_of_mass function considers the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events


def cnn(cluster_matrix, cluster_info, events, tot_only):
    # Do keras imports here, as importing earlier may raise errors unnecessary when keras will not be used
    from keras.models import load_model
    import keras

    # Hide some of the TensorFlow debug information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set amount of cores to use for TensorFlow when using CPU only
    # Set to a limit of 1 GPU
    keras.backend.set_session(
        keras.backend.tf.Session(config=keras.backend.tf.ConfigProto(intra_op_parallelism_threads=lib.config.settings.cores,
                                                                     inter_op_parallelism_threads=lib.config.settings.cores,
                                                                     device_count={'GPU': 1}
                                                                     )
                                 )
    )

    # Load model
    package_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(package_directory, 'cnn_models', lib.config.settings.event_cnn_model)

    if not os.path.exists(model_path):
        logger.error('CNN model %s does not exist.' % model_path)
        raise Exception('CNN model %s does not exist.' % model_path)

    model = load_model(model_path)

    # Delete ToA matrices, required for ToT only CNN
    if tot_only:
        cluster_matrix = np.delete(cluster_matrix, 1, 1)

    # Check model shape and input shape
    if cluster_matrix.shape[1:4] != model.layers[0].input_shape[1:4]:
        logger.error('Cluster matrix shape %s does not match model shape %s. Change cluster_matrix_size or use --event_cnn_tot_only?' % (
            cluster_matrix.shape, model.layers[0].input_shape))
        raise Exception

    # Run CNN prediction
    predictions = model.predict(cluster_matrix, batch_size=lib.config.settings.event_chunk_size, verbose=1)

    # Copy all events from cluster_info as base
    # TODO: This loads whole cluster_info matrix at once and may cause memory issues
    events = cluster_info[()].astype(dt_event)

    # Add prediction offset from cluster origin
    events['x'] = events['x'] + predictions[:, 0]
    events['y'] = events['y'] + predictions[:, 1]

    shape = tpx3format.calculate_image_shape()

    # Check for events outside matrix shape, and delete those
    ind_del = (events['x'] > shape) | (events['y'] > shape)
    indices = np.arange(len(events))
    events = np.delete(events, indices[ind_del], axis=0)
    deleted = np.count_nonzero(ind_del)

    if deleted > 0:
        logger.warning('Removed %d events found outside image matrix shape (%d).' % (deleted, shape))

    return events
