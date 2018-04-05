import logging
import os

import datetime
import h5py
import lib
import tpx3format
from lib import constants

logger = logging.getLogger('root')


class io:
    write = None

    def __init__(self):
        pass

    def open_write(self, file_name, overwrite):
        if os.path.exists(file_name) and not overwrite:
            raise IOException("Output file already exists and --overwrite not specified.")

        try:
            self.write = h5py.File(file_name, 'w')
        except IOError as e:
            raise IOException("Could not open file for writing: %s" % str(e))

    def close_write(self):
        self.write.flush()
        self.write.close()

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

    def write_hit_chunk(self, hits):
        if 'hits' not in self.write:
            if len(hits) < constants.HITS_CHUNK_SIZE:
                # It's possible we're parsing less than chunk size hits
                shape = (len(hits),)
            else:
                shape = (constants.HITS_CHUNK_SIZE,)

            self.write.create_dataset('hits', shape=shape, dtype=constants.dt_hit, maxshape=(None,), chunks=shape,
                                      data=hits)
        else:
            hits_f = self.write['hits']

            old = len(hits_f)
            hits_f.resize(old + len(hits), 0)

            hits_f[old:] = hits

        self.write.flush()

    def store_hits(self, control_events, file_name):
        self.write_base_attributes('hits')
        self.write['hits'].attrs['input_file_name'] = file_name
        self.write['hits'].attrs['shape'] = tpx3format.calculate_image_shape()

        if lib.config.settings.hits_tot_correct_file != "0":
            self.write['hits'].attrs['tot_correction_file'] = lib.config.settings.hits_tot_correct_file

        self.write['control'] = control_events
        self.write_base_attributes('control')
        self.write['control'].attrs['input_file_name'] = file_name

    def del_hits(self):
        del self.write['hits']

    def write_cluster_chunk(self, ci, cm):
        if not 'cluster_info' in self.write:
            cms = lib.config.settings.cluster_matrix_size
            self.write.create_dataset('cluster_info', shape=(len(ci),), dtype=constants.dt_ci, maxshape=(None,),
                                      chunks=(constants.CLUSTER_CHUNK_SIZE,), data=ci)
            self.write.create_dataset('clusters', shape=(len(cm), 2, cms, cms), dtype='uint8', maxshape=(None, 2, cms, cms),
                                      chunks=(constants.CLUSTER_CHUNK_SIZE, 2, cms, cms), data=cm)
        else:
            ci_f = self.write['cluster_info']
            cm_f = self.write['clusters']

            old = len(ci_f)

            ci_f.resize(old + len(ci), 0)
            cm_f.resize(old + len(cm), 0)

            ci_f[old:] = ci
            cm_f[old:] = cm

    def store_clusters(self):
        self.write_base_attributes('clusters')
        self.write_base_attributes('cluster_info')

    def del_clusters(self):
        del self.write['cluster_info']
        del self.write['clusters']

    def store_events(self, events, algorithm, cnn_model):
        self.write['events'] = events
        self.write_base_attributes('events')
        self.write['events'].attrs['algorithm'] = algorithm
        self.write['events'].attrs['shape'] = tpx3format.calculate_image_shape()

        if algorithm == 'cnn':
            self.write['events'].attrs['cnn_model'] = cnn_model

    def read_h5(self, file_name):
        if not os.path.exists(file_name):
            raise IOException("File %s for reading does not exist" % file_name)

        return h5py.File(file_name, 'r')

    def read_hits(self, file_name):
        f = self.read_h5(file_name)

        if 'hits' not in f:
            raise IOException("File %s does not have a /hits dataset" % file_name)

        return f['hits']

    def read_clusters(self, file_name):
        f = self.read_h5(file_name)

        if 'clusters' not in f:
            raise IOException("File %s does not have a /clusters dataset" % file_name)

        if 'cluster_info' not in f:
            raise IOException("File %s does not have a /cluster_info dataset" % file_name)

        return f['clusters'], f['cluster_info']


class IOException(Exception):
    pass
