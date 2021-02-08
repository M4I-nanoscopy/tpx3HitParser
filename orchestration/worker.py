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

            # TODO: Reimplement freq_tot
            # if self.settings.freq_tot:
            #     freq_tot = tpx3format.build_freq_tot(hits)
            #     io.store_freq_tot(freq_tot)

            if self.settings.C:
                clusters, cluster_info = self.parse_hits(hits)

                if self.settings.store_clusters:
                    self.store_clusters(len(hits), clusters, cluster_info)

                if self.settings.E:
                    e = self.parse_clusters(clusters, cluster_info)

                    if self.settings.store_events:
                        self.store_events(len(hits), e)

            # TODO: Figure out if this is pretty
            if not self.settings.store_hits and not self.settings.store_clusters and not self.settings.store_events:
                self.output_queue.put({'n_hits': len(hits)})

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
        cm_chunk, ci_chunk = clusters.find_clusters(self.settings, hits)

        return cm_chunk, ci_chunk

    # From clusters to events
    def parse_clusters(self, clusters, cluster_info):
        e = events.localise_events(clusters, cluster_info, self.settings.algorithm)

        return e

    def store_hits(self, hits):
        output = {
            'hits': hits,
            'n_hits': len(hits)
        }

        self.output_queue.put(output)

    def store_clusters(self, n_hits, clusters, cluster_info):
        output = {
            'clusters': clusters,
            'cluster_info': cluster_info,
            'n_hits': n_hits
        }

        self.output_queue.put(output)

    def store_events(self, n_hits, e):
        output = {
            'events': e,
            'n_hits': n_hits
        }

        self.output_queue.put(output)

