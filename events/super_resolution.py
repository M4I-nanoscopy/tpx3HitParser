import logging
import time
import numpy as np

logger = logging.getLogger('root')


def subpixel_event_redistribution(events):
    logger.info("Started event redistribution on %d events" % (len(events)))
    begin = time.time()

    # Read all events into memory
    events = events[()]

    # Make a copy of the events for writing to new file
    new_events = np.array(events, copy=True)

    # Calculate the subpixel position
    sub_x = np.mod(events['x'], np.ones(len(events)))
    sub_y = np.mod(events['y'], np.ones(len(events)))

    # Calculate the base pixel
    remain_x = events['x'] - sub_x
    remain_y = events['y'] - sub_y

    # Calculate the borders to split up evenly
    border_x = float(np.median(sub_x, axis=0))
    border_y = float(np.median(sub_y, axis=0))

    logger.info("Redistribution positions in x and y: %f02, %f02" % (border_x, border_y))

    # Redistribute
    sub_x[sub_x <= border_x] = 0.25
    sub_x[sub_x > border_x] = 0.75
    sub_y[sub_y <= border_y] = 0.25
    sub_y[sub_y > border_y] = 0.75

    # Replace back into events
    new_events['x'] = remain_x + sub_x
    new_events['y'] = remain_y + sub_y

    time_taken = time.time() - begin

    logger.info("Finished event redistribution in %d seconds ( %d events/s )" % (time_taken, len(events) / time_taken))

    return new_events
