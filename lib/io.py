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
    @staticmethod
    def git_version():
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

    def read_h5(self, file_name):
        if not os.path.exists(file_name):
            raise IOException("File %s for reading does not exist")

        return h5py.File(file_name, 'r')

    def read_hits(self, file_name):
        f = self.read_h5(file_name)

        if not 'hits' in f:
            raise IOException("File %s does not have a /hits dataset" % file_name)

        return f['hits'][()]

    def read_clusters(self, file_name):
        f = self.read_h5(file_name)

        if not 'clusters' in f:
            raise IOException("File %s does not have a /clusters dataset" % file_name)

        if not 'cluster_info' in f:
            raise IOException("File %s does not have a /cluster_info dataset" % file_name)

        return f['clusters'][()], f['cluster_info'][()]


class IOException(Exception):
    pass
