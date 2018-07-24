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

    def write_cluster_index_chunk(self, clusters):
        if 'cluster_index' not in self.write:
            self.write.create_dataset('cluster_index', shape=(len(clusters), 16), dtype='int64', maxshape=(None,16),
                                      chunks=(constants.CLUSTER_CHUNK_SIZE, 16), data=clusters)
        else:
            clusters_f = self.write['cluster_index']

            old = len(clusters_f)
            clusters_f.resize(old + len(clusters), 0)

            clusters_f[old:] = clusters

    def write_cluster_chunk(self, ci, cm):
        if 'cluster_info' not in self.write:
            cms = lib.config.settings.cluster_matrix_size
            self.write.create_dataset('cluster_info', shape=(len(ci),), dtype=constants.dt_ci, maxshape=(None,),
                                      chunks=(constants.CLUSTER_CHUNK_SIZE,), data=ci)
            self.write.create_dataset('clusters', shape=(len(cm), 2, cms, cms), dtype=constants.dt_clusters,
                                      maxshape=(None, 2, cms, cms),
                                      chunks=(constants.CLUSTER_CHUNK_SIZE, 2, cms, cms), data=cm)
        else:
            ci_f = self.write['cluster_info']
            cm_f = self.write['clusters']

            old = len(ci_f)

            ci_f.resize(old + len(ci), 0)
            cm_f.resize(old + len(cm), 0)

            ci_f[old:] = ci
            cm_f[old:] = cm

    def store_clusters(self, cluster_stats, cluster_max_sum_tot, cluster_min_sum_tot, cluster_max_size, cluster_min_size):
        self.write_base_attributes('cluster_info')
        self.write_base_attributes('clusters')

        # Store cluster_stats
        self.write.create_dataset('cluster_stats', shape=(len(cluster_stats), 2), dtype='uint16', data=cluster_stats)

        self.write['cluster_stats'].attrs.update({
            'cluster_min_sum_tot': cluster_min_sum_tot,
            'cluster_max_sum_tot': cluster_max_sum_tot,
            'cluster_min_size': cluster_min_size,
            'cluster_max_size': cluster_max_size
        })

    def store_freq_tot(self, freq_tot):
        self.write['freq_tot'] = freq_tot
        self.write_base_attributes('freq_tot')

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

        return f['clusters'], f['cluster_info']


class IOException(Exception):
    pass
