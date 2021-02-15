import logging
import struct

import h5py
import numpy as np

from lib.constants import *
import lib
import os

# TODO: Logging does not work for multiprocessing processes on Windows
logger = logging.getLogger('root')


def read_positions(f):
    control_events = []
    i = 0
    rollover_counter = 0
    approaching_rollover = False
    leaving_rollover = False
    while True:
        b = f.read(8)
        cursor = f.tell()

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

        # The SPIDR time is 16 bit (65536). It has a rollover time of 26.843 seconds
        time = pkg & 0xffff
        rollover = rollover_counter

        # Check if the time is nearing the limit of the rollover
        if time > 0.9 * 65536.:
            if leaving_rollover:
                # We have already increased the rollover counter, so we need to reset it
                rollover = rollover_counter - 1
            elif not approaching_rollover:
                # We must be approaching it
                logger.debug("Approaching SPIDR timer rollover")
                approaching_rollover = True

        # We have been approaching the rollover, so if now see a low time, it probably is a rollover
        if approaching_rollover and time < 0.01 * 65536.:
            logger.debug("SPIDR timer rollover")
            approaching_rollover = False
            leaving_rollover = True
            rollover_counter += 1
            rollover = rollover_counter

        # We are leaving the rollover, but we're far away by now
        if leaving_rollover and time > 0.1 * 65536.:
            logger.debug("Leaving SPIDR timer rollover")
            approaching_rollover = False
            leaving_rollover = False

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
                yield [cursor+16, size-(8*2), chip_nr, rollover]

        elif pkg_type == 0x6:
            pass
            logger.debug("TDC timestamp at position %d len %d" % (cursor, size))
            # TODO: Use TDC packages
            # tdc = parse_tdc_packet(pkg)
        elif pkg_type == 0xb:
            yield [cursor, size, chip_nr, rollover]
            i += 1
        else:
            logger.warning("Found packet with unknown type %d" % pkg_type)

        # Skip over the data packets and to the next header
        f.seek(cursor + size, 0)


def parse_heartbeat_packet(pkg, size):
    time = pkg >> 16
    if pkg >> 56 == 0x44:
        lsb = time & 0xffffffff
        logger.debug('Heartbeat (LSB). lsb %d. len %d' % (lsb, size))
    if pkg >> 56 == 0x45:
        msb = (time & 0xFFFFFFFF) << 32
        logger.debug('Heartbeat (MSB). msb %d. len %d' % (msb, size))
    return


# TDC (Time to Digital Converter) packages can come from the external trigger
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
        return "ToT correction file (%s) does not exists" % correct_file

    f = h5py.File(correct_file, 'r')

    if 'tot_correction' not in f:
        return "ToT correction file %s does not contain a tot_correction matrix" % correct_file

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
    return tot_correction.item((ToT, y, x, chip_id))


def apply_toa_railroad_correction_phase1_um(x, cToA, chipId):
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


def apply_toa_railroad_correction_phase1_basel(x, cToA, chipId):
    # The railroad columns for pllConfig 30
    if 193 < x < 206:
        cToA = cToA - 16

    # Chips 1, 3, 0 in Maastricht/Basel
    if chipId in (3, 0) and (x == 204 or x == 205):
        cToA = cToA + 16

    # Chips 2
    if chipId == 2 and (x == 186 or x == 187):
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


def parse_data_packages(positions, f, tot_correction, settings):
    # Allocate space for storing hits
    n_hits = sum(pos[1] // 8 for pos in positions)
    hits = np.zeros(n_hits, dtype=dt_hit)

    i = 0
    for pos in positions:
        for hit in parse_data_package(f, pos, tot_correction, settings.hits_tot_threshold, settings.hits_toa_phase_correction):
            if hit is not None:
                hits[i] = hit
                i += 1

    # There may have been hits that were not parsed (failed package), resize those empty rows away.
    hits.resize((i,), refcheck=False)

    if settings.hits_remove_cross:
        hits = remove_cross_hits(hits)

    if settings.hits_combine_chips:
        combine_chips(hits, settings.hits_cross_extra_offset)

    hits = np.sort(hits, 0, 'stable', 'ToA')

    return hits


def parse_data_package(f, pos, tot_correction, tot_threshold, toa_phase_correction):
    f.seek(pos[0])
    b = f.read(pos[1])

    # Read pixels as unsigned longs. pos[1] contains number of bytes per position. Unsigned long is 8 bytes
    struct_fmt = "<{}Q".format(pos[1] // 8)

    try:
        pixels = struct.unpack(struct_fmt, b)
    except struct.error as e:
        logger.error('Failed reading data package at position %d of file (error: %s)' % (pos[0], str(e)))
        return

    if pixels[0] >> 60 != 0xb:
        logger.error('Failed parsing data package at position %d of file' % pos[0])
        yield None
        return

    for i, pixel in enumerate(pixels):
        col = (pixel & 0x0FE0000000000000) >> 52
        super_pix = (pixel & 0x001F800000000000) >> 45
        pix = (pixel & 0x0000700000000000) >> 44

        x = int(col + pix / 4)
        y = int(super_pix + (pix & 0x3))
        spidr_time = (pixel & 0xffff)
        coarse_toa = (pixel >> (16 + 14) & 0x3fff)
        tot = int((pixel >> (16 + 4)) & 0x3ff)
        fine_toa = (pixel >> 16) & 0xf

        # Combine coarse ToA with fine ToA to form the combined ToA
        toa = int(coarse_toa << 4) - int(fine_toa)

        # Check if we would like to correct for phase shifts in the ToA values
        if toa_phase_correction > 0:
            # Shifting all cToA one full cycle forward, as I do not want to go below zero due to the correction
            toa = toa + 16

            if toa_phase_correction == 1:
                toa = apply_toa_railroad_correction_phase1_um(x, toa, pos[2])
            elif toa_phase_correction == 2:
                toa = apply_toa_railroad_correction_phase1_basel(x, toa, pos[2])
            elif toa_phase_correction == 3:
                toa = apply_toa_phase2_correction(x, toa)
                toa = apply_toa_railroad_correction_phase2(x, toa)

        # Calculate the full time by using the combined info of:
        #   * the SPIDR time (16 bit)
        #   * the (corrected) coarse toa (14 bit) and the fine toa (4 bit)
        #   * the SPIDR rollover timer (pos[3])
        global_time = int((spidr_time << 18) | toa | (pos[3] << 34))

        # Apply ToT correction matrix, when requested
        if tot_correction is not None:
            tot_correct = tot + apply_tot_correction(tot_correction, tot, y, x, pos[2])
        else:
            tot_correct = tot

        if tot_correct < tot_threshold:
            yield None
        else:
            yield pos[2], x, y, tot_correct, global_time

