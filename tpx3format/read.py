import logging
import multiprocessing
import random
import struct

import h5py
import numpy as np

from lib import constants
from lib.constants import *
import lib
from tqdm import tqdm
import os

# TODO: Logging does not work for multiprocessing processes on Windows
logger = logging.getLogger('root')


def read_raw(file_name, cores):
    f = open(file_name, "rb")
    estimate = os.fstat(f.fileno()).st_size / 8

    # Allocate an array to hold positions of packages. Using int64 to support files over 4.2 GB
    max_positions = 500
    positions = np.empty((max_positions, 3), dtype='int64')

    # Check if we have a loadable ToT correction file
    check_tot_correction(lib.config.settings.hits_tot_correct_file)

    # Allocate processing processes
    pool = multiprocessing.Pool(cores, initializer=lib.init_worker, maxtasksperchild=1000)

    # Make progress bar to keep track of hits being read
    logger.info("Reading file %s, estimating %d hits" % (file_name, estimate))
    progress_bar = tqdm(total=estimate, unit="hits", smoothing=0.1, unit_scale=True)

    def pb_update(res):
        progress_bar.update(len(res))

    control_events = []
    results = {}
    n_hits = 0
    mode = 0
    i = 0
    r = 0
    while True:
        b = f.read(8)
        position = f.tell()

        if not b:
            # Reached EOF
            break
        if len(b) < 8:
            logger.error("Truncated file, no full header at file position %d" % f.tell())
            break

        header = struct.unpack('<bbbbbbbb', b)

        chip_nr = header[4]
        mode = header[5]

        # Check for mode
        if mode != 0:
            logger.warning("Found data packet with mode %d. Code has been developed for mode 0." % mode)

        size = ((0xff & header[7]) << 8) | (0xff & header[6])

        # If this is a size 1 package, this could be control event package
        control_event = False
        if size / 8 == 1:
            control_event = parse_control_packet(f, position)

            if control_event:
                control_events.append(control_event)

        # If it is a true pixel package, add it to the list to be parsed later
        if not control_event:
            positions[i] = [position, size, chip_nr]
            i += 1

            # Chunk is ready to be processed, off load to sub process
            if i == max_positions:
                results[r] = pool.apply_async(parse_data_packages,
                                              args=[np.copy(positions), file_name, lib.config.settings],
                                              callback=pb_update)
                r += 1
                i = 0

            n_hits += size // 8

        # Break early when max_hits has been reached
        if 0 < lib.config.settings.max_hits < n_hits:
            break

        # Skip over the data packets and to the next header
        f.seek(position + size, 0)

    logger.info("File %s contains %d hits in mode %d " % (file_name, n_hits, mode))
    progress_bar.total = n_hits

    # Parse remaining bit of packages
    results[r] = pool.apply_async(parse_data_packages, args=[positions[0:i], file_name, lib.config.settings],
                                  callback=pb_update)
    pool.close()

    hits = np.empty(constants.HITS_CHUNK_SIZE, dtype=dt_hit)
    offset = 0
    parsed_hits = 0
    for idx in range(0, len(results)):
        hits_chunk = results[idx].get(timeout=100)
        parsed_hits += len(hits_chunk)

        # Fill up hits until max size, then yield
        if offset + len(hits_chunk) < len(hits):
            hits[offset:offset + len(hits_chunk)] = hits_chunk
            offset += len(hits_chunk)
        else:
            # How much more fit before yielding
            fit = len(hits) - offset

            # Store to fill up, and yield
            hits[offset:offset + fit] = hits_chunk[0:fit]
            yield hits, control_events

            # Reset
            hits = np.empty(constants.HITS_CHUNK_SIZE, dtype=dt_hit)
            offset = 0

            # Fill new chunk with remainder
            hits[0:len(hits_chunk) - fit] = hits_chunk[fit:]
            offset += len(hits_chunk) - fit

        # This reduces memory usage, by signaling the GC that this process is done
        del results[idx]

    progress_bar.close()

    # Resize remainder of hits to exact size and yield
    hits = np.resize(hits, offset)
    yield hits, control_events

    if lib.config.settings.hits_remove_cross:
        # TODO: This is an indirect way of calculating this!
        diff = n_hits - parsed_hits
        logger.info("Removed %d (%.2f percent) hits in chip border pixels or below ToT threshold (%d)"
                    % (diff, float(diff) / float(n_hits) * 100, lib.config.settings.hits_tot_threshold))


def check_tot_correction(correct_file):
    if correct_file == "0":
        # No ToT correction requested
        return True

    if not os.path.exists(correct_file):
        raise Exception("ToT correction file (%s) does not exists" % correct_file)

    f = h5py.File(correct_file, 'r')

    if 'tot_correction' not in f:
        raise Exception("ToT correction file does not contain a tot_correction matrix" % correct_file)

    data = f['tot_correction']

    logger.info("Found ToT correction file that was created on %s" % data.attrs['creation_date'])

    return True


def read_tot_correction(correct_file):
    if correct_file == "0":
        # No ToT correction requested
        return None

    f = h5py.File(correct_file, 'r')
    data = f['tot_correction']

    return data[()]


def remove_cross_hits(hits):
    # Maybe not the cleanest way to do this, but it's fast
    ind_3x = (hits['chipId'] == 3) & (hits['x'] == 255)
    ind_3y = (hits['chipId'] == 3) & (hits['y'] == 255)

    ind_0x = (hits['chipId'] == 0) & (hits['x'] == 0)
    ind_0y = (hits['chipId'] == 0) & (hits['y'] == 255)

    ind_1x = (hits['chipId'] == 1) & (hits['x'] == 255)
    ind_1y = (hits['chipId'] == 1) & (hits['y'] == 255)

    ind_2x = (hits['chipId'] == 2) & (hits['x'] == 0)
    ind_2y = (hits['chipId'] == 2) & (hits['y'] == 255)

    # Combine all found hits
    ind = ind_3x | ind_3y | ind_0x | ind_0y | ind_1x | ind_1y | ind_2x | ind_2y
    indices = np.arange(len(hits))
    hits = np.delete(hits, indices[ind], axis=0)

    return hits


def apply_tot_correction(tot_correction, ToT, y, x, chip_id):

    # Chip 0
    if chip_id == 0 and x == 0:
        return 0
    if chip_id == 0 and y == 255:
        return 0
    # Chip 1
    if chip_id == 1 and x == 255:
        return 0
    if chip_id == 1 and y == 255:
        return 0
    # # Chip 2
    if chip_id == 2 and x == 0:
        return 0
    if chip_id == 2 and y == 255:
        return 0
    # Chip 3
    if chip_id == 3 and y == 255:
        return 0
    if chip_id == 3 and x == 255:
        return 0

    return tot_correction.item((ToT, y, x, chip_id))


def calculate_image_shape():
    return 512 + 2 * lib.config.settings.hits_cross_extra_offset


def combine_chips(hits, hits_cross_extra_offset):
    # Chip are orientated like this
    # 2 1
    # 3 0

    # Calculate extra offset required for the cross pixels
    offset = 256 + 2 * hits_cross_extra_offset

    # ChipId 0
    ind = tuple([hits['chipId'] == 0])
    hits['x'][ind] = hits['x'][ind] + offset
    hits['y'][ind] = 255 - hits['y'][ind] + offset

    # ChipId 1
    ind = tuple([hits['chipId'] == 1])
    hits['x'][ind] = 255 - hits['x'][ind] + offset
    # hits['y'][ind] = hits['y'][ind]

    # ChipId 2
    ind = tuple([hits['chipId'] == 2])
    hits['x'][ind] = 255 - hits['x'][ind]
    # hits['y'][ind] = hits['y'][ind]

    # ChipId 3
    ind = tuple([hits['chipId'] == 3])
    # hits['x'][ind] = hits['x'][ind]
    hits['y'][ind] = 255 - hits['y'][ind] + offset

    # logger.debug("Combined chips to one matrix")


def parse_control_packet(f, pos):
    # Read 1 pixel packet
    f.seek(pos)
    b = f.read(8)

    struct_fmt = "<Q"
    pkg = struct.unpack(struct_fmt, b)[0]

    # Get SPIDR time and CHIP ID
    time = pkg & 0xffff
    chip_id = (pkg >> 16) & 0xffff

    if pkg >> 60 == 0xb:
        # This is NOT a event packet, but a normal pixel packet with a length of 1
        return False
    elif pkg >> 48 == CONTROL_END_OF_COMMAND:
        logger.info('EndOfCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif pkg >> 48 == CONTROL_END_OF_READOUT:
        logger.info('EndOfReadOut on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif pkg >> 48 == CONTROL_END_OF_SEQUANTIAL_COMMAND:
        logger.info('EndOfResetSequentialCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif pkg >> 48 == CONTROL_OTHER_CHIP_COMMAND:
        logger.debug('OtherChipCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    else:
        logger.debug('Unknown control packet (0x%04x) on chip ID %04x at SPIDR_TIME %5d' % (pkg >> 48, chip_id, time))

    return [pkg >> 48, chip_id, time]


def parse_data_packages(positions, file_name, settings):
    # Reopen file in new process
    f = open(file_name, "rb")

    # Allocate space for storing hits
    n_hits = sum(pos[1] // 8 for pos in positions)
    hits = np.zeros(n_hits, dtype=dt_hit)

    # Load ToT correction matrix
    tot_correction = read_tot_correction(settings.hits_tot_correct_file)

    i = 0
    for pos in positions:
        for hit in parse_data_package(f, pos, tot_correction, settings.hits_tot_threshold):
            if hit is not None:
                hits[i] = hit
                i += 1

    # There may have been hits that were not parsed (failed package), resize those empty rows away.
    hits.resize(i)

    if settings.hits_remove_cross:
        hits = remove_cross_hits(hits)

    if settings.hits_combine_chips:
        combine_chips(hits, settings.hits_cross_extra_offset)

    # TODO: Implement sorting
    # hits = np.sort(hits, 0, 'heapsort', 'TSPIDR')

    return hits


def parse_data_package(f, pos, tot_correction, tot_threshold):
    f.seek(pos[0])
    b = f.read(pos[1])

    # Read pixels as unsigned longs. pos[1] contains number of bytes per position. Unsigned long is 8 bytes
    struct_fmt = "<{}Q".format(pos[1] // 8)

    try:
        pixels = struct.unpack(struct_fmt, b)
    except struct.error as e:
        logger.error('Failed reading data data package at position %d of file (error: %s)' % (pos[0], str(e)))
        return

    time = pixels[0] & 0xffff

    if pixels[0] >> 60 == 0xb and pos[2] < 4:
        for i, pixel in enumerate(pixels):
            dcol = (pixel & 0x0FE0000000000000) >> 52
            spix = (pixel & 0x001F800000000000) >> 45
            pix = (pixel & 0x0000700000000000) >> 44

            x = dcol + pix / 4
            y = spix + (pix & 0x3)

            ToA = (pixel >> (16 + 14)) & 0x3fff
            ToT = (pixel >> (16 + 4)) & 0x3ff
            FToA = (pixel >> 16) & 0xf
            CToA = (ToA << 4) | (~FToA & 0xf)

            # Apply ToT correction matrix, when requested
            if tot_correction is not None:
                ToT_correct = int(ToT) + apply_tot_correction(tot_correction, int(ToT), int(y), int(x), pos[2])
            else:
                ToT_correct = ToT

            if ToT_correct < tot_threshold:
                yield None
            else:
                yield (pos[2], x, y, ToT_correct, CToA, time)
    else:
        logger.error('Failed parsing data package at position %d of file' % pos[0])
        yield None
