## Getting ready

Download

```
git clone https://github.com/M4I-nanoscopy/tpx3HitParser.git
cd tpx3HitParser
```

Recommended way is to use a Python virtualenv. But this is optional.

Python 2.7 (not recommended)
```
virtualenv tpx3-py27
source tpx3-py27/bin/activate
```

Python 3
```
python3 -m venv tpx3-py3
source tpx3-py3/bin/activate
```

Install Python dependencies

```
pip install -r requirements.txt
```

## Running

```
./tpx3Hitparser.py --help
optional arguments:
  -h, --help                     show this help message and exit
  -c FILE, --config FILE         Specify other config file (default: None)
  -v, --verbose                  Verbose output (default: False)

input arguments:
  --raw FILE                     Read raw .tpx3 (default: None)
  --hits FILE                    Read .h5 file containing /hits (default: None)
  --clusters FILE                Read .h5 file containing /clusters (default: None)
  --events FILE                  Read .h5 file containing /events (default: None)

parse arguments:
  -C                             Parse clusters (default: False)
  -E                             Parse events (default: False)

output arguments:
  -o FILE, --output FILE         Output HDF5 file (default: None)
  --overwrite                    Overwrite existing HDF5 file (default: False)
  --store_hits                   Store /hits in output file (default: False)
  --store_clusters               Store /clusters in output file (default: False)
  --store_cluster_indices        Store /cluster_index in output file (for determining Delta ToA
                                 correction) (default: False)
  --store_events                 Store /events in output file (default: False)
  --store_predictions            Store /predictions in output file (default: False)

correct arguments:
  --correct_super_res            Correct and redistribute super resolution events (default: False)
  --correct_chip_edge            Correct chip edge events (default: False)

miscellaneous arguments:
  --cluster_stats                Store cluster stats (default: True)
  --freq_tot                     Parse and store ToT frequency matrix (default: False)
  --freq_toa                     Parse and store delta-ToA frequency matrix (default: False)

constants:
  --cores N                      Number of cores to use (default: 4)
  --max_hits N                   Maximum number of hits to read (0: infinite) (default: 0)
  --hits_remove_cross 0/1        Remove the middle border pixels between the chips (default: True)
  --hits_combine_chips 0/1       Combine the chips to one matrix (default: True)
  --hits_cross_extra_offset N    Extra offset used for the cross pixels per chip when combining the
                                 chips (default: 2)
  --hits_tot_correct_file FILE   ToT correction file, or 0 for no correction (default: 0)
  --hits_ftoa_correct_file FILE  ToT correction file, or 0 for no correction (default: 0)
  --hits_toa_phase_correction N  Apply ToA correction. 0=None, 1=Maastricht-Pll30, 2=Basel-Pll30,
                                 3=Pll94 (default: 0)
  --hits_tot_threshold N         Below this ToT threshold hits are not stored (default: 5)
  --cluster_time_window N        Maximum time interval between individual hits to cluster them (in
                                 cToA values) (default: 50)
  --cluster_min_size N           Minimum cluster size (default: 2)
  --cluster_max_size N           Maximum cluster size (default: 10)
  --cluster_max_sum_tot N        Maximum cluster sum tot (default: 400)
  --cluster_min_sum_tot N        Minimum cluster sum tot (default: 200)
  --cluster_chunk_size N         Number of hits to consider at once (memory intensive!) (default:
                                 6000)
  --cluster_matrix_size N        Size of the resulting cluster matrix (default: 10)
  --event_cnn_model FILE         CNN model to use for event localisation (default: model-200kv-
                                 tottoa.h5)
  --event_cnn_tot_only 0/1       The specified CNN model uses ToT only (default: False)
  -a A, --algorithm A            Event localisation algorithm to use (default: centroid)
```

## Configuration

All options are command line options. Defaults for constants are specified in a separate config file `default.cfg`. You can either
edit this file or make your own and specify this with the `--config` option.


## Copyright

(c) Maastricht University

## License

MIT License

## Authors

Paul van Schayck (p.vanschayck@maastrichtuniversity.nl)