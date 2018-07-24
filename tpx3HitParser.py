#!/usr/bin/env python

import sys
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
    try:
        io.open_write(settings.output, settings.overwrite)
    except lib.IOException as e:
        logger.error(str(e))
        return 1

    # Hits ###
    hits = None
    if settings.raw:
        control_events = None

        # TODO: Not pretty to return control_events in each iteration
        for hits_chunk, control_events in tpx3format.read_raw(settings.raw, settings.cores):
            io.write_hit_chunk(hits_chunk)

        # Store hits, we may need to remove it later though if not requested to store
        io.store_hits(control_events, settings.raw)

        # Read hits from just written data set, not loaded in memory. But read chunk-wise upon request
        hits = io.read_hits(settings.output)
    elif settings.hits:
        try:
            hits = io.read_hits(settings.hits)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    if settings.spidr_stats and hits is not None:
        frames.spidr_time_stats(hits)

    if settings.freq_tot:
        freq_tot = tpx3format.build_freq_tot(hits)
        io.store_freq_tot(freq_tot)

    # Clusters ###
    cluster_matrix = None
    cluster_info = None
    cluster_stats = []
    if settings.C:
        for cm_chunk, ci_chunk, index_chunk, stats_chunk in clusters.find_clusters(hits):
            io.write_cluster_chunk(ci_chunk, cm_chunk)

            # if requested store also cluster_indices
            if settings.store_cluster_indices:
                io.write_cluster_index_chunk(index_chunk)

            cluster_stats.extend(stats_chunk)

        # Store clusters and cluster stats, we may delete it later
        io.store_clusters(cluster_stats, settings.cluster_max_sum_tot, settings.cluster_min_sum_tot,
                          settings.cluster_max_size, settings.cluster_min_size)

        # Read clusters from just written data, not loaded in memory
        cluster_matrix, cluster_info = io.read_clusters(settings.output)
    elif settings.clusters:
        try:
            cluster_matrix, cluster_info = io.read_clusters(settings.clusters)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    # Events ###
    if settings.E:
        # TODO: This writes all events at once, and may cause memory issues
        e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)

        if settings.store_events:
            io.store_events(e, settings.algorithm, settings.event_cnn_model)

    if settings.raw and not settings.store_hits:
        io.del_hits()

    if settings.C and not settings.store_clusters:
        io.del_clusters()

    io.close_write()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
