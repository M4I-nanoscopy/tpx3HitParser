import os
import queue
import signal
from multiprocessing import Queue, Event, JoinableQueue, shared_memory
import numpy

from time import sleep
import logging

from tqdm import tqdm

import tpx3format
from lib.constants import tot_correction_shape
from orchestration.gpu import Gpu
from orchestration.worker import Worker
from orchestration.writer import Writer


class Orchestrator:
    writer = None
    gpu = None
    workers = []
    progress_bar = None

    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger('root')

        # Build Queues to handle the input to the workers, and the output from the workers to the writer
        self.input_queue = JoinableQueue()
        self.gpu_queue = JoinableQueue()
        self.output_queue = JoinableQueue()
        self.finished_queue = Queue()

        # Use event to signal when to stop processing
        self.keep_processing = Event()
        self.keep_processing.set()

        # Use event to signal to writer we want finalise
        self.finalise_writing = Event()

        # Use shared memory to store the ToT correction data
        self.tot_correction_shared = None

        signal.signal(signal.SIGINT, self.sigint)

    def orchestrate(self):
        # Put the ToT correction array in shared memory
        if self.settings.hits_tot_correct_file != "0":
            self.read_tot_correction()

        # Build workers to process the input
        for i in range(self.settings.cores):
            p = Worker(self.settings, self.keep_processing, self.input_queue, self.output_queue, self.gpu_queue, self.tot_correction_shared)
            p.start()
            self.workers.append(p)

        # Build GPU worker if needed
        if self.settings.algorithm == 'cnn' and self.settings.E:
            p = Gpu(self.settings, self.keep_processing, self.gpu_queue, self.output_queue)
            p.start()
            self.gpu = p

        # Build writer
        p = Writer(self.settings, self.keep_processing, self.finalise_writing, self.output_queue, self.finished_queue)
        p.start()
        self.writer = p

        # Open main input file
        f = open(self.settings.raw, "rb")
        estimate = os.fstat(f.fileno()).st_size / 8

        # Make progress bar to keep track of hits being read
        self.logger.info("Reading file %s, estimating %d hits" % (self.settings.raw, estimate))
        self.progress_bar = tqdm(total=estimate, unit="hits", smoothing=0.1, unit_scale=True)

        positions = []
        n_hits = 0
        n_chunks = 0
        chunk_hits = 0
        for position in tpx3format.read_positions(f):
            positions.append(position)

            chunk_hits += position[1] // 8
            n_hits += position[1] // 8

            # Break early when max_hits has been reached
            if 0 < self.settings.max_hits < n_hits:
                break

            if chunk_hits > self.settings.cluster_chunk_size:
                n_chunks += 1
                self.input_queue.put(positions)
                positions = []
                chunk_hits = 0

        # We don't need the input file anymore in this process
        f.close()

        # Push remainder to input queue
        n_chunks += 1
        self.input_queue.put(positions)

        self.logger.debug("File %s contains %d hits" % (self.settings.raw, n_hits))
        self.progress_bar.total = n_hits

        processed_chunks = 0
        while self.keep_processing.is_set() and processed_chunks < n_chunks:
            try:
                hits_done = self.finished_queue.get(timeout=1)
            except queue.Empty:
                continue

            processed_chunks += 1
            self.progress_bar.update(hits_done)

        # Signal to the write we want to finalise the output file
        self.finalise_writing.set()
        self.writer.join()

        # Finished!
        self.progress_bar.close()
        self.logger.info("Finished")

    def read_tot_correction(self):
        # Store the the ToT correction in shared memory buffer, between the processes
        tc = tpx3format.read_tot_correction(self.settings.hits_tot_correct_file)
        self.tot_correction_shared = shared_memory.SharedMemory(create=True, size=tc.nbytes)
        tc_shared = numpy.ndarray(tot_correction_shape, dtype=tc.dtype, buffer=self.tot_correction_shared.buf)
        tc_shared[:] = tc[:]

    def sigint(self, signum, frame):
        # Start signalling to child processes that we're terminating early
        self.keep_processing.clear()
        # self.cleanup() is being called by the finally of the caller

    def cleanup(self):
        # Close progress bar
        self.progress_bar.close()

        # Allow workers to finish their job
        self.logger.debug("Waiting 2s for workers to stop...")
        sleep(2)
        for worker in self.workers:
            worker.terminate()
            worker.join()

        # Writer
        self.writer.terminate()
        self.writer.join()

        # Gpu
        if self.gpu is not None:
            self.gpu.terminate()
            self.gpu.join()

        # Cancel joining queues threads. They may not be empty, but we don't care anymore at this point
        self.input_queue.cancel_join_thread()
        self.gpu_queue.cancel_join_thread()
        self.output_queue.cancel_join_thread()
        self.finished_queue.cancel_join_thread()

        # Close the shared memory
        if self.tot_correction_shared is not None:
            self.tot_correction_shared.unlink()



