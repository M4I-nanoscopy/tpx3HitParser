#!/usr/bin/env python

import sys
import h5py
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
            logger.error(e.message)
            return 1

    # Hits
    hits = []
    control_events = []
    hits_input_file = ""

    if settings.raw:
        hits, control_events = tpx3format.read_raw(settings.raw, settings.cores)
        hits_input_file = settings.raw

    if settings.hits:
        f = h5py.File(settings.hits, 'r')
        hits = f['hits'][()]
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
        f = h5py.File(settings.clusters, 'r')
        cluster_matrix = f['clusters'][()]
        cluster_info = f['cluster_info'][()]

    if settings.store_clusters:
        io.store_clusters(cluster_matrix, cluster_info)

    # Events
    if settings.E:
        e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)

        if settings.store_events:
            io.store_events(e, settings.algorithm)


    if settings.output:
        io.close_write()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
