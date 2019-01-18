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

    # TODO: This is not entirely pretty or elegant, and does not scale to 4 times super resolution

    # Calculate the borders to split up evenly
    border_x = float(np.median(sub_x, axis=0))

    # Calculate how many events we're reassigning in x
    if border_x > 0.5:
        wrong_x = np.logical_and(0.5 < sub_x, sub_x < border_x)
    else:
        wrong_x = np.logical_and(0.5 > sub_x, sub_x > border_x)

    # Calculate how many events in each X quadrant we're reassigning in Y
    sub_y_1st = sub_y[sub_x < border_x]
    border_y_1st = float(np.median(sub_y_1st, axis=0))
    sub_y_2nd = sub_y[sub_x > border_x]
    border_y_2nd = float(np.median(sub_y_2nd, axis=0))

    logger.info("Border x %.2f and y1 %.2f and y2 %.2f" % (border_x, border_y_1st, border_y_2nd))

    if border_y_1st > 0.5:
        wrong_y_1st = np.logical_and(0.5 < sub_y, sub_y < border_y_1st)
    else:
        wrong_y_1st = np.logical_and(0.5 > sub_y, sub_y > border_y_1st)

    if border_y_2nd > 0.5:
        wrong_y_2nd = np.logical_and(0.5 < sub_y, sub_y < border_y_2nd)
    else:
        wrong_y_2nd = np.logical_and(0.5 > sub_y, sub_y > border_y_2nd)

    # Calculate total amount of events re-assigning
    wrong = np.logical_or(wrong_x, wrong_y_1st, wrong_y_2nd).sum()
    logger.info("Reassigning %d events (%.2f percent)" % (wrong, (float(wrong) / float(len(events)) * float(100))))

    # Reassign per quadrant
    q1 = np.logical_and(sub_x < border_x, sub_y < border_y_1st)
    sub_x[q1] = 0.25
    sub_y[q1] = 0.25

    q2 = np.logical_and(sub_x > border_x, sub_y < border_y_2nd)
    sub_x[q2] = 0.75
    sub_y[q2] = 0.25

    q3 = np.logical_and(sub_x < border_x, sub_y > border_y_1st)
    sub_x[q3] = 0.25
    sub_y[q3] = 0.75

    q4 = np.logical_and(sub_x > border_x, sub_y > border_y_2nd)
    sub_x[q4] = 0.75
    sub_y[q4] = 0.75

    # Replace back into events
    new_events['x'] = remain_x + sub_x
    new_events['y'] = remain_y + sub_y

    time_taken = time.time() - begin

    logger.info("Finished event redistribution in %d seconds ( %d events/s )" % (time_taken, len(events) / time_taken))

    return new_events
