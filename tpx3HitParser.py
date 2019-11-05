#!/usr/bin/env python

import sys
import lib
import clusters
import tpx3format
import events
import logging

logger = lib.setup_custom_logger('root', logging.INFO)

def main():
    # Parse all command line arguments
    lib.parse_config()
    settings = lib.config.settings

    if settings.verbose:
        log_level = logging.DEBUG
        logger.setLevel(log_level)

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
        io.store_clusters(cluster_stats, settings.cluster_time_window, settings.cluster_max_sum_tot, settings.cluster_min_sum_tot,
                          settings.cluster_max_size, settings.cluster_min_size)

        # Read clusters from just written data, not loaded in memory
        cluster_matrix, cluster_info = io.read_clusters(settings.output)
    elif settings.clusters:
        try:
            cluster_matrix, cluster_info = io.read_clusters(settings.clusters)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    if settings.freq_toa:
        cluster_indices = None

        # We need to read hits as well when using cluster indices. Maybe not pretty to read this here again, but alas
        if settings.C:
            cluster_indices = io.read_cluster_indices(settings.output)
            hits = io.read_hits(settings.output)
        elif settings.clusters:
            cluster_indices = io.read_cluster_indices(settings.clusters)
            hits = io.read_hits(settings.clusters)

        freq_toa = clusters.build_freq_toa(cluster_indices, hits)
        io.store_freq_toa(freq_toa)

    # Events ###
    e = None
    if settings.E:
        # TODO: This writes all events at once, and may cause memory issues
        e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)
    elif settings.events:
        try:
            e = io.read_events(settings.events)
        except lib.IOException as e:
            logger.error(str(e))
            return 1

    # Post event parsing corrections ###

    # Super Resolution
    if settings.correct_super_res:
        e = events.subpixel_event_redistribution(e)

    # Correct chip edges
    if settings.correct_chip_edge:
        e = events.chip_edge_correct(e)

    if settings.raw and not settings.store_hits:
        io.del_hits()

    if settings.C and not settings.store_clusters:
        io.del_clusters()

    if settings.store_events:
        io.store_events(e, settings.algorithm, settings.event_cnn_model)

    if settings.store_predictions:
        io.store_predictions(events.calculate_predictions(e, cluster_info), settings.algorithm)

    io.close_write()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except lib.UserConfigException as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
