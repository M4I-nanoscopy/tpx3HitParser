import logging
import queue
import signal
from multiprocessing import Process
import lib


class Writer(Process):
    min_toa = -1
    max_toa = -1

    def __init__(self, settings, keep_processing, finalise_writing, oq, fq):
        Process.__init__(self)

        self.settings = settings
        self.output_queue = oq
        self.finished_queue = fq
        self.keep_processing = keep_processing
        self.finalise_writing = finalise_writing
        self.logger = logging.getLogger('root')
        self.io = lib.io()

    def run(self):
        # Ignore the interrupt signal. Let parent handle that.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Open output file
        try:
            self.io.open_write(self.settings.output, self.settings.overwrite)
        except lib.IOException as e:
            self.logger.error(str(e))
            return 1

        while self.keep_processing.is_set():
            try:
                data = self.output_queue.get(timeout=1)
            except queue.Empty:
                # Check if the orchestrator is signalling there is no more output data coming
                if self.finalise_writing.is_set():
                    break
                else:
                    continue

            if 'min_toa' in data and 'max_toa' in data:
                self.min_max_toa(data['min_toa'], data['max_toa'])

            if self.settings.store_hits and 'hits' in data:
                self.write_hits(data['hits'])

            if self.settings.store_clusters and 'clusters' in data:
                self.write_clusters(data['cluster_info'], data['clusters'])

            if self.settings.store_events and 'events' in data:
                self.write_events(data['events'])

            # When the data is final (intermediate==True), we should signal orchestrator we're done
            if not data['intermediate']:
                self.finished_queue.put({
                    'n_hits': data['n_hits'],
                    'chunks': data['chunks'] if 'chunks' in data else 1
                })

            self.output_queue.task_done()

        # Finish writing, and close file
        self.finalise()

    def min_max_toa(self, min_toa, max_toa):
        if min_toa > 0:
            if self.min_toa == -1:
                self.min_toa = min_toa
            else:
                self.min_toa = min(self.min_toa, min_toa)

        if max_toa > 0:
            if self.max_toa == -1:
                self.max_toa = max_toa
            else:
                self.max_toa = max(self.max_toa, max_toa)

    def finalise(self):
        if self.settings.store_hits:
            self.io.store_hits(self.settings.raw, self.settings.hits_tot_correct_file,
                               self.settings.hits_cross_extra_offset, self.min_toa, self.max_toa)

        if self.settings.store_clusters:
            self.io.store_clusters(self.settings.cluster_time_window, self.settings.cluster_max_sum_tot,
                                   self.settings.cluster_min_sum_tot,
                                   self.settings.cluster_max_size, self.settings.cluster_min_size)

        if self.settings.store_events:
            self.io.store_events(self.settings.algorithm, self.settings.event_cnn_model,
                                 self.settings.hits_cross_extra_offset, self.min_toa, self.max_toa, self.settings.event_correct_chip_edges)

        self.io.close_write()

    def write_hits(self, hits):
        self.io.write_hit_chunk(hits)

    def write_clusters(self, cluster_info, clusters):
        self.io.write_cluster_chunk(cluster_info, clusters, self.settings.cluster_matrix_size, self.settings.cluster_stats)

    def write_events(self, e):
        self.io.write_event_chunk(e, self.settings.cluster_stats)

