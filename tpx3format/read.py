import logging
import multiprocessing
import struct
import numpy as np
from lib.constants import *

logger = logging.getLogger('root')

f_name = ""


def read_raw(file_name, cores):
    global f_name
    f_name = file_name

    f = file(f_name, "rb")

    positions = []
    control_events = []
    n_hits = 0
    mode = 0
    while True:
        b = f.read(8)

        if not b:
            # Reached EOF
            break
        if len(b) < 8:
            logger.error("Truncated file, no full header at file position %d" % f.tell())
            break

        header = struct.unpack('<bbbbbbbb', b)

        chip_nr = header[4]
        mode = header[5]
        size = ((0xff & header[7]) << 8) | (0xff & header[6])

        # If this is a size 1 package, this could be control event package
        control_event = False
        if size / 8 == 1:
            control_event = parse_control_packet(f, f.tell())

            if control_event:
                control_events.append(control_event)

        # If it is a true pixel package, add it to the list to be parsed later
        if not control_event:
            positions.append([f.tell(), size, chip_nr])
            n_hits += size / 8

        f.seek(size, 1)

    hits = np.empty((n_hits, 8), 'uint16')

    logger.info("File %s contains %d hits in mode %d " % (f_name, n_hits, mode))

    pool = multiprocessing.Pool(cores)

    chunk_size = len(positions) / cores
    slice = chunks(positions, chunk_size)

    results = []
    for i, chunk in enumerate(slice):
        results.append(pool.apply_async(parse_data_packages, args=[chunk]))
    pool.close()

    offset = 0
    for r in results:
        hits_chunk = r.get(timeout=999999)

        hits[offset:offset + len(hits_chunk)] = hits_chunk
        offset += len(hits_chunk)

    return hits, control_events


def parse_control_packet(f, pos):
    # Read package and reverse position (is this a clean way?)
    f.seek(pos)
    b = f.read(8)
    f.seek(pos)

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


def parse_data_packages(positions):
    global f_name

    # Reopen file in new process
    f = file(f_name, "rb")

    n_hits = sum(pos[1] / 8 for pos in positions)
    hits = np.empty((n_hits, 8), 'uint16')

    i = 0
    for pos in positions:
        for hit in parse_data_package(f, pos):
            if hit is not None:
                hits[i] = hit
                i += 1

    return hits


def parse_data_package(f, pos):
    f.seek(pos[0])
    b = f.read(pos[1])

    # Read pixels as unsigned longs. pos[1] contains number of bytes per position. Unsigned long is 8 bytes
    struct_fmt = "<{}Q".format(pos[1] / 8)
    pixels = struct.unpack(struct_fmt, b)

    time = pixels[0] & 0xffff

    if pixels[0] >> 60 == 0xb and pos[2] < 4:
        for i, pixel in enumerate(pixels):
            dcol = (pixel & 0x0FE0000000000000L) >> 52
            spix = (pixel & 0x001F800000000000L) >> 45
            pix = (pixel & 0x0000700000000000L) >> 44

            x = dcol + pix / 4
            y = spix + (pix & 0x3)

            ToA = (pixel >> (16 + 14)) & 0x3fff
            ToT = (pixel >> (16 + 4)) & 0x3ff
            FToA = (pixel >> 16) & 0xf
            CToA = (ToA << 4) | (~FToA & 0xf)

            yield [pos[2], x, y, ToT, ToA, FToA, time, CToA]
    else:
        logger.error('Failed parsing data package')
        yield None


def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]
