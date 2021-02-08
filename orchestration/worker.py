import logging
import signal
from multiprocessing import Process
import queue

import clusters
import events
import tpx3format


class Worker(Process):
    input_queue = None
    output_queue = None

    def __init__(self, settings, keep_processing, iq, oq):
        # Setting this process to daemon, makes them killed if the child is killed.
        Process.__init__(self)

        self.input_queue = iq
        self.output_queue = oq
        self.settings = settings
        self.keep_processing = keep_processing
        self.logger = logging.getLogger('root')

    def run(self):
        # Ignore the interrupt signal. Let parent (orchestrator) handle that.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.f = open(self.settings.raw, "rb")

        while self.keep_processing.is_set():
            try:
                positions = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            hits = self.parse_raw(positions)

            if self.settings.store_hits:
                self.store_hits(hits)

            # if self.settings.freq_tot:
            #     freq_tot = tpx3format.build_freq_tot(hits)
            #     io.store_freq_tot(freq_tot)

            # TODO: Figure out if this is pretty
            if not self.settings.store_hits and not self.settings.store_clusters and not self.settings.store_events:
                self.output_queue.put({'n_hits': len(hits)})
            # clusters = self.parse_clusters(hits)

            self.input_queue.task_done()

        # Cleanup
        self.f.close()

    def terminate(self):
        Process.terminate(self)

    # From raw to hits
    def parse_raw(self, positions):
        # Hits ###
        hits_chunk = tpx3format.parse_data_packages(positions, self.f, self.settings)
        return hits_chunk

    # From hits to clusters
    def parse_hits(self, hits):
        pass
        # Clusters
        # cluster_matrix = None
        # cluster_info = None
        # cluster_stats = []
        # if self.settings.C:
        # cm_chunk, ci_chunk, index_chunk, stats_chunk = clusters.find_clusters(hits)
        # cluster_stats.extend(stats_chunk)

        # Store clusters and cluster stats, we may delete it later
        # io.store_clusters(cluster_stats, settings.cluster_time_window, settings.cluster_max_sum_tot,
        #                   settings.cluster_min_sum_tot,
        #                   settings.cluster_max_size, settings.cluster_min_size)

    # From clusters to events
    def parse_clusters(self, clusters):
        pass
        # Events ###
        # e = None
        # if self.settings.E:
        #     e = events.localise_events(cluster_matrix, cluster_info, settings.algorithm)

    def store_hits(self, hits):
        output = {
            'hits': hits,
            'n_hits': len(hits)
        }

        self.output_queue.put(output)
