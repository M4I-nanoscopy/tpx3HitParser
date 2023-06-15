import logging
import os

import datetime
import h5py
from tpx3hitparser import lib
from tpx3hitparser import tpx3format
from tpx3hitparser.lib import constants

from tpx3hitparser import clusters
import tpx3hitparser.events as ev

logger = logging.getLogger('root')

class io:
    write = None

    def __init__(self):
        pass

    def check_write(self, file_name, overwrite):
        if os.path.exists(file_name) and not overwrite:
            return "Output file %s already exists and --overwrite not specified." % file_name
        else:
            return True

    def open_write(self, file_name, overwrite=False, append=False):
        if os.path.exists(file_name) and not (overwrite or append):
            raise lib.IOException("Output file already exists and --overwrite not specified.")

        if append:
            mode = 'a'
        else:
            mode = 'w'

        try:
            self.write = h5py.File(file_name, mode)
        except IOError as e:
            raise lib.IOException("Could not open file for writing: %s" % str(e))

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
            shape = (constants.HITS_CHUNK_SIZE,)
            self.write.create_dataset('hits', dtype=constants.dt_hit, maxshape=(None,), chunks=shape, data=hits)
        else:
            hits_f = self.write['hits']
            old = len(hits_f)
            hits_f.resize(old + len(hits), 0)
            hits_f[old:] = hits

    def store_hits(self, file_name, hits_tot_correct_file, hits_cross_extra_offset, min_toa, max_toa):
        if 'hits' not in self.write:
            logger.warning("There was no dataset /hits written to the output file. Was nothing processed?")
            return

        self.write_base_attributes('hits')
        self.write['hits'].attrs['input_file_name'] = file_name
        self.write['hits'].attrs['min_toa'] = min_toa
        self.write['hits'].attrs['max_toa'] = max_toa
        self.write['hits'].attrs['shape'] = tpx3format.calculate_image_shape(hits_cross_extra_offset)

        if hits_tot_correct_file != "0":
            self.write['hits'].attrs['tot_correction_file'] = hits_tot_correct_file

    def replace_hits(self, hits):
        self.write['hits'][...] = hits

    def del_hits(self):
        del self.write['hits']

    def write_cluster_chunk(self, ci, cm, cms, cluster_stats):
        if 'cluster_info' not in self.write:
            self.write.create_dataset('cluster_info', shape=(len(ci),), dtype=clusters.cluster_info_datatype(cluster_stats), maxshape=(None,),
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

    def store_clusters(self, cluster_time_window, cluster_max_sum_tot, cluster_min_sum_tot, cluster_max_size, cluster_min_size):
        if 'clusters' not in self.write:
            logger.warning("There was no dataset /clusters written to the output file. Was nothing processed?")
            return

        self.write_base_attributes('clusters')

        self.write['clusters'].attrs.update({
            'cluster_time_window': cluster_time_window,
            'cluster_min_sum_tot': cluster_min_sum_tot,
            'cluster_max_sum_tot': cluster_max_sum_tot,
            'cluster_min_size': cluster_min_size,
            'cluster_max_size': cluster_max_size
        })

    def store_freq_tot(self, freq_tot):
        self.write['freq_tot'] = freq_tot
        self.write_base_attributes('freq_tot')

    def store_freq_toa(self, freq_toa):
        self.write['freq_toa'] = freq_toa
        self.write_base_attributes('freq_toa')

    def del_clusters(self):
        del self.write['cluster_info']
        del self.write['clusters']

    def write_event_chunk(self, events, cluster_stats):
        if 'events' not in self.write:
            shape = (constants.EVENTS_CHUNK_SIZE,)
            self.write.create_dataset('events', dtype=ev.event_info_datatype(cluster_stats), maxshape=(None,), chunks=shape, data=events)
        else:
            events_f = self.write['events']
            old = len(events_f)
            events_f.resize(old + len(events), 0)
            events_f[old:] = events

    def store_events(self, algorithm, cnn_model, hits_cross_extra_offset, min_toa, max_toa, event_correct_chip_edges):
        if 'events' not in self.write:
            logger.warning("There was no dataset /events written to the output file. Was nothing processed?")
            return

        self.write_base_attributes('events')
        self.write['events'].attrs['algorithm'] = algorithm
        self.write['events'].attrs['min_toa'] = min_toa
        self.write['events'].attrs['max_toa'] = max_toa
        self.write['events'].attrs['shape'] = ev.calculate_image_shape(hits_cross_extra_offset, event_correct_chip_edges)

        if algorithm == 'cnn':
            self.write['events'].attrs['cnn_model'] = cnn_model

    def replace_events(self, events):
        self.write['events'][...] = events

    def store_predictions(self, predictions, algorithm):
        name = '/predictions/%s' % algorithm
        self.write.create_dataset(name, data=predictions)
        self.write_base_attributes(name)

    def read_h5(self, file_name):
        if not os.path.exists(file_name):
            raise lib.IOException("File %s for reading does not exist" % file_name)

        return h5py.File(file_name, 'r')

    def read_hits(self, file_name):
        f = self.read_h5(file_name)

        if 'hits' not in f:
            raise lib.IOException("File %s does not have a /hits dataset" % file_name)

        return f['hits']

    def read_clusters(self, file_name):
        f = self.read_h5(file_name)

        if 'clusters' not in f:
            raise lib.IOException("File %s does not have a /clusters dataset" % file_name)

        return f['clusters'], f['cluster_info']

    def read_events(self, file_name):
        f = self.read_h5(file_name)

        if 'events' not in f:
            raise lib.IOException("File %s does not have a /events dataset" % file_name)

        return f['events']

