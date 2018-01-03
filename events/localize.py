import logging
import time
from scipy import ndimage
from lib.constants import *
import numpy as np
import os

logger = logging.getLogger('root')


def localize_events(cluster_matrix, cluster_info, method):
    logger.info("Started event localization on %d events using method %s" % (len(cluster_info), method))
    begin = time.time()

    events = np.empty((len(cluster_info), 4), 'float64')

    if method == "centroid":
        centroid(cluster_matrix, cluster_info, events)
    elif method == "cnn":
        cnn(cluster_matrix, cluster_info, events)

    time_taken = time.time() - begin

    logger.info(
        "Finished event localization in %d seconds ( %d events/s )" % (time_taken, len(cluster_info) / time_taken))

    return events


def centroid(cluster_matrix, cluster_info, events):
    for idx, cluster in enumerate(cluster_matrix):
        # Center of mass of ToT cluster
        x, y = ndimage.measurements.center_of_mass(cluster[0])

        events[idx][E_CHIP] = cluster_info[idx][CI_CHIP]
        events[idx][E_X] = cluster_info[idx][CI_X] + x + 0.5
        events[idx][E_Y] = cluster_info[idx][CI_Y] + y + 0.5
        events[idx][E_TIME] = cluster_info[idx][CI_cTOA]

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
    pred = model.predict(x_test, batch_size=(10^6))

    for idx, p in enumerate(pred):
        events[idx][E_CHIP] = cluster_info[idx][CI_CHIP]
        events[idx][E_X] = cluster_info[idx][CI_X] + p[0]
        events[idx][E_Y] = cluster_info[idx][CI_Y] + p[1]
        events[idx][E_TIME] = cluster_info[idx][CI_cTOA]

    return events
