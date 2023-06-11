import logging
import signal
from multiprocessing import Process
import queue
import numpy

from tpx3hitparser import clusters
from tpx3hitparser import events
from tpx3hitparser import tpx3format
from tpx3hitparser.lib.constants import TOT_CORRECTION_SHAPE


class Worker(Process):
    def __init__(self, settings, keep_processing, iq, oq, gq, tc):
        # Setting this process to daemon, makes them killed if the child is killed.
        Process.__init__(self)

        self.input_queue = iq
        self.output_queue = oq
        self.gpu_queue = gq
        self.settings = settings
        self.keep_processing = keep_processing
        self.logger = logging.getLogger('root')

        self.f = None
        self.tot_correction_shared = tc
        self.tot_correction = None

    def run(self):
        # Ignore the interrupt signal. Let parent (orchestrator) handle that.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.f = open(self.settings.raw, "rb")

        while self.keep_processing.is_set():
            try:
                positions = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            hits, min_toa, max_toa = self.parse_raw(positions)

            output = {
                'n_hits': len(hits),
                'intermediate': False,
                'min_toa': min_toa,
                'max_toa': max_toa
            }

            if self.settings.store_hits:
                output['hits'] = hits

            if self.settings.C:
                cl, cl_info = self.parse_hits(hits)

                if self.settings.store_clusters:
                    if self.settings.store_clusters:
                        output['clusters'] = cl
                        output['cluster_info'] = cl_info

                if self.settings.E:
                    if self.settings.algorithm != 'cnn':
                        e = self.parse_clusters(cl, cl_info)

                        if self.settings.store_events:
                            output['events'] = e
                    else:
                        # Send of to dedicated GPU process
                        self.gpu_queue.put({'clusters': cl, 'cluster_info': cl_info, 'n_hits': len(hits)})
                        # Need to signal to write class that this chunk is not finished yet
                        output['intermediate'] = True

            self.output_queue.put(output)
            self.input_queue.task_done()

        # Cleanup
        self.f.close()

    def terminate(self):
        Process.terminate(self)

    # From raw to hits
    def parse_raw(self, positions):
        # Load ToT correction matrix from shared memory
        if self.tot_correction is None and self.settings.hits_tot_correct_file != "0":
            self.tot_correction = numpy.ndarray(TOT_CORRECTION_SHAPE, dtype=numpy.int16, buffer=self.tot_correction_shared.buf)

        hits_chunk, min_toa, max_toa = tpx3format.parse_data_packages(positions, self.f, self.tot_correction, self.settings)

        return hits_chunk, min_toa, max_toa

    # From hits to clusters
    def parse_hits(self, hits):
        cm_chunk, ci_chunk = clusters.find_clusters(self.settings, hits)

        return cm_chunk, ci_chunk

    # From clusters to events
    def parse_clusters(self, cl, cluster_info):
        e = events.localise_events(cl, cluster_info, self.settings.algorithm, self.settings.cluster_stats)

        if self.settings.event_correct_chip_edges:
            e = events.chip_edge_correct(e)

        return e
