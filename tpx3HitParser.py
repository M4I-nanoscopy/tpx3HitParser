#!/usr/bin/env python

import sys
import h5py
import lib
import frames
import clusters
import tpx3format
import events

logger = lib.setup_custom_logger('root')


def main():
    config = lib.config()
    logger.debug(config)

    if config.output:
        w = h5py.File(config.output, 'w')

    if config.raw:
        hits = tpx3format.read_raw(config.raw, config.cores)

        if config.store_hits:
            w['hits'] = hits

    if config.hits:
        f = h5py.File(config.hits, 'r')
        hits = f['hits'][()]

    if config.frame_hits:
        frame = frames.build(hits)

        if config.stats:
            frames.stats(frame)

        if config.show:
            frames.show(frame)

    if config.spidr_stats:
        frames.spidr_time_stats(hits)

    if config.clusters:
        f = h5py.File(config.clusters, 'r')
        cluster_matrix = f['clusters'][()]
        cluster_info = f['cluster_info'][()]

    if config.C:
        cluster_info, cluster_matrix = clusters.find_clusters(hits, config.cores)

        if config.store_clusters:
            w['clusters'] = cluster_matrix
            w['cluster_info'] = cluster_info

    if config.E:
        e = events.localize_events(cluster_matrix, cluster_info, config.algorithm)

        if config.store_events:
            w['events'] = e

    if config.output:
        w.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)