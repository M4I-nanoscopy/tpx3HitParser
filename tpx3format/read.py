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

    # Check if we have a loadable fToA correction file
    check_ftoa_correction(lib.config.settings.hits_ftoa_correct_file)

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
            logger.error("Header packet with mode %d. Code has been developed for mode 0." % mode)

        size = ((0xff & header[7]) << 8) | (0xff & header[6])

        # Read the first package of the data package to figure out its type
        pkg = struct.unpack("<Q", f.read(8))[0]
        pkg_type = pkg >> 60

        # Parse the different package types
        if pkg_type == 0x7:
            control_event = parse_control_packet(pkg, size)

            if control_event:
                control_events.append(control_event)
        elif pkg_type == 0x4:
            parse_heartbeat_packet(pkg, size)

            # TODO: Use heartbeat packages in calculating time

            # Heartbeat packages are always followed by a 0x7145 or 0x7144 control package, and then possibly
            # pixels. Continue to parse those pixels, but strip away the control package
            if size - (8*2) > 0:
                positions[i] = [position+16, size-(8*2), chip_nr]

        elif pkg_type == 0x6:
            logger.info("TDC timestamp at position %d len %d" % (position, size))
            # TODO: Use TDC packages
            # tdc = parse_tdc_packet(pkg)
        elif pkg_type == 0xb:
            positions[i] = [position, size, chip_nr]
            i += 1
        else:
            logger.warning("Found packet with unknown type %d" % pkg_type)

        # Check if chunk is ready to be processed, off load to sub process
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


def parse_heartbeat_packet(pkg, size):
    if pkg >> 56 == 0x44:
        lsb = (pkg >> 16) & 0xffffffff
        logger.debug('Heartbeat (LSB). lsb %d. len %d' % (lsb, size))
    if pkg >> 56 == 0x45:
        msb = (pkg >> 16) & 0xff
        logger.debug('Heartbeat (MSB). msb %d. len %d' % (msb, size))
    return


def parse_tdc_packet(pkg):
    tdc_type = pkg >> 56
    counter = (pkg >> 44) & 0xfff
    timestamp = (pkg >> 9) & 0x3ffffffff
    stamp = (pkg >> 4) & 0xf

    logger.debug("TDC package. Type: 0x%04x. Counter: %d. Timestamp: %d. Stamp: %d" % (tdc_type, counter, timestamp, stamp))

    return


def parse_control_packet(pkg, size):
    # Get SPIDR time and CHIP ID
    time = pkg & 0xffff
    chip_id = (pkg >> 16) & 0xffff

    control_type = pkg >> 48

    if size / 8 > 1:
        logger.warning("Control data packet is followed by more data. This is unexpected")

    if control_type == CONTROL_END_OF_COMMAND:
        logger.info('EndOfCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif control_type == CONTROL_END_OF_READOUT:
        logger.info('EndOfReadOut on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif control_type == CONTROL_END_OF_SEQUANTIAL_COMMAND:
        logger.info('EndOfResetSequentialCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    elif control_type == CONTROL_OTHER_CHIP_COMMAND:
        logger.debug('OtherChipCommand on chip ID %04x at SPIDR_TIME %5d' % (chip_id, time))
    else:
        logger.debug('Unknown control packet (0x%04x) on chip ID %04x at SPIDR_TIME %5d' % (pkg >> 48, chip_id, time))

    return [control_type, chip_id, time]


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


def check_ftoa_correction(correct_file):
    if correct_file == "0":
        # No fToA correction requested
        return True

    if not os.path.exists(correct_file):
        raise Exception("fToA correction file (%s) does not exists" % correct_file)

    f = h5py.File(correct_file, 'r')

    if 'corrector' not in f:
        raise Exception("fToA correction file does not contain a ftoa_correction matrix" % correct_file)

    #data = f['ftoa_correction']
    #   logger.info("Found fToA correction file that was created on %s" % data.attrs['creation_date'])

    return True


def read_tot_correction(correct_file):
    if correct_file == "0":
        # No ToT correction requested
        return None

    f = h5py.File(correct_file, 'r')
    data = f['tot_correction']

    return data[()]


def read_ftoa_correction(correct_file):
    if correct_file == "0":
        # No fToA correction requested
        return None

    f = h5py.File(correct_file, 'r')
    data = {
        'corrector': f['corrector'][()],
        'classList': f['classList'][()],
        'dctoa_shift' : f['dctoa_shift'][()]
    }

    return data


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


def apply_toa_railroad_correction_phase1(x, cToA, chipId):
    # The railroad columns for pllConfig 30
    if 193 < x < 206:
        cToA = cToA - 16

    # Chips 2, 3, 0 in Maastricht/Basel
    if chipId in (2, 3, 0) and (x == 204 or x == 205):
        cToA = cToA + 16

    # Chips 1 in Maastricht/Basel
    if chipId == 1 and (x == 186 or x == 187):
        cToA = cToA - 16

    return cToA


def apply_toa_railroad_correction_phase2(x, cToA):
    # The railroad columns for pllConfig 94
    if x == 196 or x == 197 or x == 200 or x == 201 or x == 204 or x == 205:
        cToA = cToA - 16

    return cToA


def apply_toa_phase2_correction(x, cToA):
    # PHASE 2 (pllConfig 94)
    if int(x % 4) == 2 or int(x % 4) == 3:
        cToA = cToA - 8

    return cToA

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


def parse_data_packages(positions, file_name, settings):
    # Reopen file in new process
    f = open(file_name, "rb")

    # Allocate space for storing hits
    n_hits = sum(pos[1] // 8 for pos in positions)
    hits = np.zeros(n_hits, dtype=dt_hit)

    # Load ToT correction matrix
    tot_correction = read_tot_correction(settings.hits_tot_correct_file)

    # Load fToA correction matrix
    ftoa_correction = read_ftoa_correction(settings.hits_ftoa_correct_file)

    i = 0
    for pos in positions:
        for hit in parse_data_package(f, pos, tot_correction, settings.hits_tot_threshold, settings.hits_toa_phase_correction, ftoa_correction):
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


def parse_data_package(f, pos, tot_correction, tot_threshold, toa_phase_correction, ftoa_correction):
    f.seek(pos[0])
    b = f.read(pos[1])

    # Read pixels as unsigned longs. pos[1] contains number of bytes per position. Unsigned long is 8 bytes
    struct_fmt = "<{}Q".format(pos[1] // 8)

    try:
        pixels = struct.unpack(struct_fmt, b)
    except struct.error as e:
        logger.error('Failed reading data package at position %d of file (error: %s)' % (pos[0], str(e)))
        return

    time = pixels[0] & 0xffff

    if pixels[0] >> 60 == 0xb and pos[2] < 4:
        for i, pixel in enumerate(pixels):
            dcol = (pixel & 0x0FE0000000000000) >> 52
            spix = (pixel & 0x001F800000000000) >> 45
            pix = (pixel & 0x0000700000000000) >> 44

            x = int(dcol + pix / 4)
            y = int(spix + (pix & 0x3))
            ToA = int((pixel >> (16 + 14)) & 0x3fff)
            ToT = int((pixel >> (16 + 4)) & 0x3ff)
            fToA = int((pixel >> 16) & 0xf)
            spId = int(dcol / 2) * 64 + int(spix / 4)

            # Combine coarse ToA (ToA) with fine ToA (fToA) to form the combined ToA (cToA)
            #CToA = (ToA << 4) | (~fToA & 0xf)

            if ftoa_correction is not None:
                sp_class = int(ftoa_correction['classList'][spId]) - 1

                # 16, 8, 4, 12
                length_ftoa = ftoa_correction['corrector'][fToA, pix, 2, sp_class]
                end_ftoa = ftoa_correction['corrector'][fToA, pix, 3, sp_class]

                try:
                    fToA = random.randint(end_ftoa - length_ftoa, end_ftoa - 1)
                except ValueError:
                    fToA = 0

                CToA = ToA * 160 - fToA

                # dCToA Shift
                # CToA = ToA * 160 - fToA*10 - int(ftoa_correction['dctoa_shift'][y, x] * 10)

                # cToA phase shift due to fToA pattern
                # shift = ftoa_correction['corrector'][15, pix, 2, sp_class] - 10
                # CToA += shift

            if toa_phase_correction:
                # Shifting all cToA one full cycle forward, as I do not want to go below zero due to the correction
                CToA = CToA + 16

                # CToA = apply_toa_phase2_correction(x, CToA)
                # CToA = apply_toa_railroad_correction_phase2(x, CToA)
                CToA = apply_toa_railroad_correction_phase1(x, CToA, pos[2])

            # Apply ToT correction matrix, when requested
            if tot_correction is not None:
                ToT_correct = int(ToT) + apply_tot_correction(tot_correction, ToT, y, x, pos[2])
            else:
                ToT_correct = ToT

            if ToT_correct < tot_threshold:
                yield None
            else:
                yield (pos[2], x, y, ToT_correct, CToA, time, fToA, spId, int(pix))
    else:
        logger.error('Failed parsing data package at position %d of file' % pos[0])
        yield None
