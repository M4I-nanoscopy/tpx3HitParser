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

    # Print config statements
    logger.debug(settings)

    if settings.output:
        w = h5py.File(settings.output, 'w')

    if settings.raw:
        hits, control_events = tpx3format.read_raw(settings.raw, settings.cores)

        if settings.store_hits:
            w['hits'] = hits
            w['control'] = control_events

    if settings.hits:
        f = h5py.File(settings.hits, 'r')
        hits = f['hits'][()]

    if settings.frame_hits:
        frame = frames.build(hits)

        if settings.stats:
            frames.stats(frame)

        frames.show(frame)

    if settings.spidr_stats:
        frames.spidr_time_stats(hits)

    if settings.clusters:
        f = h5py.File(settings.clusters, 'r')
        cluster_matrix = f['clusters'][()]
        cluster_info = f['cluster_info'][()]

    if settings.C:
        cluster_info, cluster_matrix = clusters.find_clusters(hits)

        if settings.store_clusters:
            w['clusters'] = cluster_matrix
            w['cluster_info'] = cluster_info

    if settings.E:
        e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)

        if settings.store_events:
            w['events'] = e

    if settings.output:
        w.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
