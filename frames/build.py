import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from lib.constants import *

logger = logging.getLogger('root')


def build(hits):
    # TODO: Handle exposure time and multiple frames
    # frames = list()
    # if exposure is not None:
    #     start_time = d[0][SPIDR_TIME]
    #     frames = split(d, start_time, exposure)
    #
    # for events in frames:
    #

    frame = hits_to_frame(hits)
    return frame


# This function converts the event data to a 256 by 256 matrix and places the chips into a full frame
# Also deals with the chip positions
def hits_to_frame(frame):
    # TODO the data type here is int8, may not be appropriate
    img = np.zeros(shape=(512, 512), dtype=np.int32)

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = frame[[frame[:, CHIP] == chip]]

        rows = chip_events[:, X]
        cols = chip_events[:, Y]
        data = np.ones(len(rows))
        #data = chip_events[:, TOT]

        # This is a very fast way to construct the 256, 256 matrix from the event based data
        # Much faster than looping
        chip_frame = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(256, 256))

        # TODO: Chips position need to be configurable
        if chip == 0:
            img[256:512, 256:512] = np.rot90(chip_frame.todense(), k=1)
        if chip == 1:
            img[0:256, 256:512] = np.rot90(chip_frame.todense(), k=-1)
        if chip == 2:
            img[0:256, 0:256] = np.rot90(chip_frame.todense(), k=-1)
        if chip == 3:
            img[256:512, 0:256] = np.rot90(chip_frame.todense(), k=1)

    return img


# This function yields frames from the main list, so is memory efficient
def split(d, start_time, exposure):
    frames = [list(), list(), list(), list(), list()]
    last_frame = 0
    frame_offset = 0

    # TODO: Here we read only one chunk, this may be good place to multi thread??
    events = d[0:d.chunks[0]]

    for event in events:
        # The SPIDR_TIME has about a 20s timer, it may reset mid run
        if event[SPIDR_TIME] < start_time:
            logger.debug("SPIDR_TIME reset to %d, from %d" % (event[SPIDR_TIME], start_time))
            start_time = event[SPIDR_TIME]
            frame_offset = last_frame + len(frames)

        # Calculate current frame
        frame = frame_offset + int((event[SPIDR_TIME] - start_time) / exposure)

        if frame < last_frame:
            logger.warn("Wrong order of events! %i - %d" % (frame, last_frame))
            continue

        # Yield previous frame if we are above buffer space of frames list
        if frame >= last_frame + 5:
            yield frames.pop(0)
            # Add empty frame
            frames.append(list())
            # Increase counter
            last_frame = last_frame + 1

        frames[frame - last_frame].append(event)

    # Yield remainder of frames
    for frame in frames:
        yield frame
