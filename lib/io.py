import logging
import os

import datetime
import h5py
import lib
from lib import constants

logger = logging.getLogger('root')


class io:
    write = None
    read = None
    amend = False

    def __init__(self):
        pass

    def open_read(self, file_name):
        # TODO: Implement reading of files
        self.read = h5py.File(file_name, 'r')

    def open_write(self, file_name, overwrite, amend):
        if os.path.exists(file_name) and not (overwrite or amend):
            raise IOException("Output file already exists and --overwrite or --amend not specified.")

        if amend:
            mode = 'a'
            self.amend =True
        else:
            mode = 'w'

        try:
            self.write = h5py.File(file_name, mode)
        except IOError as e:
            raise IOException("Could not open file for writing: %s" % e.message)

    def close_write(self):
        self.write.close()

    # Return the git revision as a string
    def git_version(self):
        try:
            out = lib.minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            git_revision = out.strip().decode('ascii')

            if git_revision == "":
                git_revision = "Unknown"
        except OSError:
            git_revision = "Not determined"

        return git_revision

    def write_base_attributes(self, dataset):
        d = self.write[dataset]

        d.attrs['revision'] = self.git_version()
        d.attrs['date'] = datetime.datetime.now().isoformat()
        d.attrs['version'] = constants.VERSION

    def store_hits(self, hits, control_events, file_name):

        if self.amend and 'hits' in self.write:
            logger.warn('Overwriting existing hits dataset')
            del self.write['hits']
        if self.amend and 'control' in self.write:
            logger.warn('Overwriting existing control dataset')
            del self.write['control']

        self.write['hits'] = hits
        self.write_base_attributes('hits')
        self.write['hits'].attrs['input_file_name'] = file_name
        self.write['control'] = control_events
        self.write_base_attributes('control')
        self.write['control'].attrs['input_file_name'] = file_name

    def store_clusters(self, cluster_matrix, cluster_info):
        # TODO: Implement amend

        self.write['clusters'] = cluster_matrix
        self.write_base_attributes('clusters')
        self.write['cluster_info'] = cluster_info
        self.write_base_attributes('cluster_info')

    def store_events(self, events, algorithm, cnn_model):
        # TODO: Implement amend

        self.write['events'] = events
        self.write_base_attributes('events')
        self.write['events'].attrs['algorithm'] = algorithm

        if algorithm == 'cnn':
            self.write['events'].attrs['cnn_model'] = cnn_model


class IOException(Exception):
    pass
