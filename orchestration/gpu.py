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
        self.chunks = []

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
                data = self.gpu_queue.get(timeout=2)
            except queue.Empty:
                # The queue is starving, we're most likely near the end, parse anything we still have saved up
                self.logger.debug("GPU queue starved")
                self.parse_clusters_gpu()
                continue

            self.chunks.append(data['n_hits'])

            # Bunch (and parse if needed) clusters
            self.bunch_clusters(data['clusters'], data['cluster_info'])

            self.gpu_queue.task_done()

    # Group clusters together, because it's suboptimal to do event localisation on small batches
    def bunch_clusters(self, cluster_chunk, cluster_info_chunk):
        if self.offset + len(cluster_chunk) < EVENTS_CHUNK_SIZE:
            self.clusters[self.offset:self.offset + len(cluster_chunk)] = cluster_chunk
            self.cluster_info[self.offset:self.offset + len(cluster_chunk)] = cluster_info_chunk
            self.offset += len(cluster_chunk)
        else:
            # First fill up remainder of clusters, and then send this off to parse
            remainder = EVENTS_CHUNK_SIZE - self.offset
            self.clusters[self.offset:] = cluster_chunk[0:remainder]
            self.cluster_info[self.offset:] = cluster_info_chunk[0:remainder]
            self.offset += remainder

            # Parse
            self.parse_clusters_gpu()

            # Store left over of the chunk in a new part
            new_chunk = len(cluster_chunk) - remainder
            self.clusters[0:new_chunk] = cluster_chunk[remainder:]
            self.cluster_info[0:new_chunk] = cluster_info_chunk[remainder:]
            self.offset = new_chunk

    # Parse and sent to GPU
    def parse_clusters_gpu(self):
        e = events.cnn(self.clusters[:self.offset], self.cluster_info[:self.offset], self.model, self.settings.event_cnn_tot_only)
        # e = events.localise_events(self.clusters[:self.offset], self.cluster_info[:self.offset], 'centroid')

        if self.settings.store_events:
            self.output_queue.put({
                'events': e,
                'chunks': len(self.chunks),
                'n_hits': sum(self.chunks),
                'intermediate': False
            })
        else:
            self.output_queue.put({
                'chunks': len(self.chunks),
                'n_hits': sum(self.chunks),
                'intermediate': False
            })

        # Reset
        self.offset = 0
        self.chunks = []

