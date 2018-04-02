#!/usr/bin/env python

import sys

# Get rid of a harmless h5py FutureWarning. Can be removed with a new release of h5py
# https://github.com/h5py/h5py/issues/961
import warnings
warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype from .*',)

import lib
import frames
import clusters
import tpx3format
import events
import logging


def main():
    # Parse all command line arguments
    lib.parse_config()
    settings = lib.config.settings

    if settings.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = lib.setup_custom_logger('root', log_level)
    io = lib.io()

    # Print config statements
    logger.debug(settings)

    # Output file
    if settings.output:
        try:
            io.open_write(settings.output, settings.overwrite, settings.amend)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    # Hits
    control_events = []
    hits_input_file = ""

    if settings.raw:
        hits_input_file = settings.raw

        # Yielding 1M hits for temp storage
        for hits in tpx3format.read_raw(settings.raw, settings.cores):
            io.write_hit_chunk(hits)

    exit(0)

    if settings.hits:
        try:
            hits = io.read_hits(settings.hits)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

        hits_input_file = settings.hits

    if settings.store_hits:
        io.store_hits(hits, control_events, hits_input_file)

    if settings.spidr_stats:
        frames.spidr_time_stats(hits)

    # Clusters
    cluster_info = []
    cluster_matrix = []
    if settings.C:
        cluster_info, cluster_matrix = clusters.find_clusters(hits)

    if settings.clusters:
        try:
            cluster_matrix, cluster_info = io.read_clusters(settings.clusters)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    if settings.store_clusters:
        io.store_clusters(cluster_matrix, cluster_info)

    # Events
    if settings.E:
        e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)

        if settings.store_events:
            io.store_events(e, settings.algorithm, settings.event_cnn_model)

    if settings.output:
        io.close_write()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
