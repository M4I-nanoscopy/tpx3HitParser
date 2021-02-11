import logging
import os
import signal
from multiprocessing import Process
import queue

import events
import numpy as np

from lib.constants import EVENTS_CHUNK_SIZE, dt_ci, dt_clusters


class Gpu(Process):
    def __init__(self, settings, keep_processing, gq, oq):
        # Setting this process to daemon, makes them killed if the child is killed.
        Process.__init__(self)

        self.gpu_queue = gq
        self.output_queue = oq
        self.settings = settings
        self.keep_processing = keep_processing
        self.logger = logging.getLogger('root')

        self.model = None
        self.clusters = np.zeros((EVENTS_CHUNK_SIZE, 2, self.settings.cluster_matrix_size, self.settings.cluster_matrix_size), dtype=dt_clusters)
        self.cluster_info = np.zeros(EVENTS_CHUNK_SIZE, dtype=dt_ci)
        self.offset = 0

    def run(self):
        # Ignore the interrupt signal. Let parent (orchestrator) handle that.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Hide some of the TensorFlow debug information
        if self.settings.verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Do keras and tensorflow imports here, as importing earlier may raise errors unnecessary
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        # Set amount of cores to use for TensorFlow when using CPU only
        tf.config.threading.set_intra_op_parallelism_threads(self.settings.cores)

        # Load model
        self.model = load_model(self.settings.event_cnn_model)

        while self.keep_processing.is_set():
            try:
                data = self.gpu_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self.settings.E and self.settings.algorithm == 'cnn':
                e = self.parse_clusters_gpu(data['clusters'], data['cluster_info'])

                if self.settings.store_events and e is not None:
                    self.store_events(data['n_hits'], e)
                else:
                    self.output_queue.put({'n_hits': data['n_hits']})

            self.gpu_queue.task_done()

    # From clusters to events
    def parse_clusters_gpu(self, cluster_chunk, cluster_info_chunk):
        # TODO: We're not parsing the remainder of clusters here!
        # Group clusters together, because it's suboptimal to do event localisation on small batches
        if self.offset + len(cluster_chunk) < EVENTS_CHUNK_SIZE:
            self.clusters[self.offset:self.offset + len(cluster_chunk)] = cluster_chunk
            self.cluster_info[self.offset:self.offset + len(cluster_chunk)] = cluster_info_chunk
            self.offset += len(cluster_chunk)

            return None
        else:
            e = events.cnn(self.clusters, self.cluster_info, self.model, self.settings.event_cnn_tot_only)
            self.offset = 0
            self.clusters[0:len(cluster_chunk)] = cluster_chunk
            self.cluster_info[0:len(cluster_chunk)] = cluster_info_chunk

            return e

    def store_events(self, n_hits, e):
        output = {
            'events': e,
            'n_hits': n_hits
        }

        self.output_queue.put(output)
