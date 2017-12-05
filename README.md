## Getting ready

Download

```
git clone https://github.com/M4I-nanoscopy/tpx3HitParser.git
cd tpx3HitParser
```

Recommended way is to use a Python virtualenv.  But this is optional.

```
virtualenv tpx3
source tpx3/bin/activate
```

Install Python dependencies

```
pip install -r requirements.txt
```

## Running

```
./tpx3Hitparser.py --help
```

## Configuration

All options are command line options. Defaults for constants are specified in a separate config file `default.cfg`. You can either
edit this file or make your own and specify this with the `--config` option.


## Copyright

(c) Maastricht University

## License

All rights reserved

## Authors

Paul van Schayck (p.vanschayck@maastrichtuniversity.nl)