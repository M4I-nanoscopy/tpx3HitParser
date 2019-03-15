import logging
import time
import numpy as np

logger = logging.getLogger('root')


def chip_edge_correct(events):
    logger.info("Started correcting %d chip edge events" % (len(events)))
    begin = time.time()

    # Read all events into memory
    events = events[()]

    # Make a copy of the events for writing to new file
    e = np.array(events, copy=True)
    ex = e['x']
    ey = e['y']

    # Shift x and y coordinate by 4 pixels for events right of the edge
    ex[ex > 257] = ex[ex > 257] + 4
    ey[ey > 257] = ey[ey > 257] + 4

    # Take second horizontal row and redivide over three pixels
    x_edge1 = np.logical_and(ex > 256, ex < 257)
    ex[x_edge1] = ex[x_edge1] + np.random.random_integers(2, 4, len(ex[x_edge1]))

    # Take first horizontal row and redivide over three pixels
    x_edge2 = np.logical_and(ex > 255, ex < 256)
    ex[x_edge2] = ex[x_edge2] + np.random.random_integers(0, 2, len(ex[x_edge2]))

    # Take second vertical column and redivide over three pixels
    y_edge1 = np.logical_and(ey > 256, ey < 257)
    ey[y_edge1] = ey[y_edge1] + np.random.random_integers(2, 4, len(ey[y_edge1]))

    # Take second vertical column and redivide over three pixels
    y_edge2 = np.logical_and(ey > 255, ey < 256)
    ey[y_edge2] = ey[y_edge2] + np.random.random_integers(0, 2, len(ey[y_edge2]))

    # Place back new coordinates in event
    e['x'] = ex
    e['y'] = ey

    time_taken = time.time() - begin

    logger.info("Finished chip edge correction in %d seconds ( %d events/s )" % (time_taken, len(events) / time_taken))

    return e
