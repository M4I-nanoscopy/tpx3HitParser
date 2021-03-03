import logging
import random
from scipy import ndimage

import tpx3format
#from lib import UserConfigException
from lib.constants import *
import numpy as np
from tqdm import tqdm

# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0

logger = logging.getLogger('root')


def localise_events(cluster_matrix, cluster_info, method, settings):
    logger.debug("Started event localization on %d events using method %s" % (len(cluster_info), method))

    if method == "centroid":
        events = calculate_centroid(cluster_matrix, cluster_info, settings)
    elif method == "random":
        events = calculate_random(cluster_matrix, cluster_info, settings)
    elif method == "highest_toa":
        events = calculate_toa(cluster_matrix, cluster_info, settings)
    elif method == "highest_tot":
        events = calculate_tot(cluster_matrix, cluster_info, settings)
    else:
        raise Exception("Chosen localisation algorithm ('%s') does not exist" % method)

    return events

def calculate_centroid(cluster_matrix, cluster_info, settings):
    events = np.empty(len(cluster_info), event_info_datatype(settings.event_stats))

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

        events[idx]['ToA'] = cluster_info[idx]['ToA']

        if settings.event_stats:
            events[idx]['sumToT'] = np.sum(cluster_matrix[idx][0])
            events[idx]['nHits'] = np.count_nonzero(cluster_matrix[idx][0])

    return events


def calculate_random(cluster_matrix, cluster_info, settings):
    events = np.empty(len(cluster_info), event_info_datatype(settings.event_stats))

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

        events[idx]['ToA'] = cluster_info[idx]['ToA']

        if settings.event_stats:
            events[idx]['sumToT'] = np.sum(cluster_matrix[idx][0])
            events[idx]['nHits'] = np.count_nonzero(cluster_matrix[idx][0])

    return events


def calculate_toa(cluster_matrix, cluster_info, settings):
    events = np.empty(len(cluster_info), event_info_datatype(settings.event_stats))

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

        events[idx]['ToA'] = cluster_info[idx]['ToA']

        if settings.event_stats:
            events[idx]['sumToT'] = np.sum(cluster_matrix[idx][0])
            events[idx]['nHits'] = np.count_nonzero(cluster_matrix[idx][0])

    return events


def calculate_tot(cluster_matrix, cluster_info, settings):
    events = np.empty(len(cluster_info), event_info_datatype(settings.event_stats))

    for idx, cluster in enumerate(cluster_matrix):
        i = np.argmax(cluster[0])
        y, x = np.unravel_index(i, cluster[0].shape)

        events[idx]['chipId'] = cluster_info[idx]['chipId']

        # We considered the coordinate of the pixel as the origin. Shift this to the middle
        # of the pixel by adding 0.5
        events[idx]['x'] = cluster_info[idx]['x'] + x + 0.5
        events[idx]['y'] = cluster_info[idx]['y'] + y + 0.5

        events[idx]['ToA'] = cluster_info[idx]['ToA']

        if settings.event_stats:
            events[idx]['sumToT'] = np.sum(cluster_matrix[idx][0])
            events[idx]['nHits'] = np.count_nonzero(cluster_matrix[idx][0])

    return events


def cnn(cluster_matrix, cluster_info, model, tot_only, hits_cross_extra_offset):
    # Delete ToA matrices, required for ToT only CNN
    if tot_only:
        cluster_matrix = np.delete(cluster_matrix, 1, 1)

    # Check model shape and input shape
    if cluster_matrix.shape[1:4] != model.layers[0].input_shape[1:4]:
        logger.error(
            'Cluster matrix shape %s does not match model shape %s. Change cluster_matrix_size or use --event_cnn_tot_only?' % (
                cluster_matrix.shape, model.layers[0].input_shape))
        raise UserConfigException

    # Run CNN prediction
    predictions = model.predict(cluster_matrix, batch_size=EVENTS_CHUNK_SIZE, verbose=0)

    # Copy all events from cluster_info as base
    events = cluster_info[()].astype(dt_event)

    # Add prediction offset from cluster origin
    events['x'] = events['x'] + predictions[:, 1]
    events['y'] = events['y'] + predictions[:, 0]

    shape = tpx3format.calculate_image_shape(hits_cross_extra_offset)

    # Check for events outside matrix shape, and delete those
    ind_del = (events['x'] > shape) | (events['y'] > shape)
    indices = np.arange(len(events))
    events = np.delete(events, indices[ind_del], axis=0)
    deleted = np.count_nonzero(ind_del)

    if deleted > 0:
        logger.debug('Removed %d events found outside image matrix shape (%d).' % (deleted, shape))

    return events

def event_info_datatype(event_stats):
    dt = dt_event_base

    if event_stats:
        dt = dt + dt_event_extended

    return numpy.dtype(dt)
