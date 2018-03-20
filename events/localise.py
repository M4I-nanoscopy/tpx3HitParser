import logging
import multiprocessing
import time
from scipy import ndimage
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
        centroid(cluster_matrix, cluster_info, events)
    elif method == "cnn":
        cnn(cluster_matrix, cluster_info, events)

    time_taken = time.time() - begin

    logger.info(
        "Finished event localization in %d seconds ( %d events/s )" % (time_taken, len(cluster_info) / time_taken))

    return events


def centroid(cluster_matrix, cluster_info, events):
    # Setup pool
    pool = multiprocessing.Pool(lib.config.settings.cores, initializer=lib.init_worker, maxtasksperchild=100)
    results = list()

    # Progress bar
    progress_bar = tqdm(total=len(cluster_info), unit="clusters", smoothing=0.1, unit_scale=True)

    # First split clusters in chunks
    chunk_size = lib.config.settings.event_chunk_size
    if chunk_size > len(cluster_info):
        logger.warning("Cluster chunk size is larger than amount of events")
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
            logger.warning("Could not calculate center of mass: empty cluster. Cluster_info: %s" % cluster_info[idx])
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
    # Do keras imports here, as importing earlier may raise errors unnecessary when keras will not be used
    from keras.models import load_model
    from keras import backend as K

    # Hide some of the TensorFlow debug information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set amount of cores to use for TensorFlow when using CPU only
    # Set to a limit of 1 GPU
    K.set_session(
        K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=lib.config.settings.cores,
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

    # Check model shape and input shape
    if cluster_matrix.shape[1:4] != model.layers[0].input_shape[1:4]:
        logger.error('Cluster matrix shape %s does not match model shape %s. Change cluster_matrix_size?' % (cluster_matrix.shape, model.layers[0].input_shape))
        raise Exception

    # Run CNN prediction
    predictions = model.predict(cluster_matrix, batch_size=lib.config.settings.event_chunk_size, verbose=1)

    # TODO: use predict_on_batch and roll our own batches. This allows us to, while processing to
    # already fill the event matrix
    for idx, p in enumerate(predictions):
        events[idx]['chipId'] = cluster_info[idx]['chipId']
        events[idx]['x'] = cluster_info[idx]['x'] + p[0]
        events[idx]['y'] = cluster_info[idx]['y'] + p[1]
        events[idx]['cToA'] = cluster_info[idx]['cToA']
        events[idx]['TSPIDR'] = cluster_info[idx]['TSPIDR']

    return events
