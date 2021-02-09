import logging
import os
import signal
from multiprocessing import Process
import queue

import events
import lib
import numpy as np

from lib.constants import EVENTS_CHUNK_SIZE, dt_ci, dt_clusters, dt_event


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
        self.n_hits = 0

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
        model_path = self.settings.event_cnn_model

        # TODO: Move this to tpx3HitParser
        if not os.path.exists(model_path):
            raise lib.UserConfigException('CNN model %s does not exist.' % model_path)

        self.model = load_model(model_path)

        while self.keep_processing.is_set():
            predictions = self.model.predict(self.yield_clusters_from_queue, steps=10, verbose=0)

            # Copy all events from cluster_info as base
            e = self.cluster_info.astype(dt_event)

            # Add prediction offset from cluster origin
            e['x'] = e['x'] + predictions[:, 1]
            e['y'] = e['y'] + predictions[:, 0]

            if self.settings.store_events:
                self.store_events(self.n_hits, e)
            else:
                self.output_queue.put({'n_hits': self.n_hits})

            self.logger.info("Parsed events")
            self.cluster_info.fill(0)

    def yield_clusters_from_queue(self):
            data = self.gpu_queue.get(block=False)

            ci = data['cluster_info']
            c = data['clusters']
            self.cluster_info[self.offset:self.offset + len(ci)] = ci
            self.offset += len(ci)
            self.n_hits += data['n_hits']

            # Delete ToA matrices, required for ToT only CNN
            if self.settings.tot_only:
                c = np.delete(c, 1, 1)

            self.gpu_queue.task_done()

            yield c

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

