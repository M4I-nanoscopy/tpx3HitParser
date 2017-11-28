import ConfigParser
import argparse
import sys


def config(argv=None):
    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
        "exposure": None,
        "frames": False,
        "cores": 4
    }

    if args.config:
        config = ConfigParser.SafeConfigParser()
        config.read([args.config])
        defaults.update(dict(config.items("Defaults")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
    )
    parser.set_defaults(**defaults)

    # Input arguments
    parser.add_argument("--raw", help="Read raw .tpx3")
    parser.add_argument("--hits", help="Read .h5 file containing /hits")
    parser.add_argument("--clusters", help="Read .h5 file containing /clusters")

    # Parse options
    parser.add_argument("-C", action='store_true', help="Parse clusters")
    parser.add_argument("-E", action='store_true', help="Parse events")

    # Output file arguments
    parser.add_argument("-o", "--output", help='Output file')
    parser.add_argument("--store_hits", action='store_true', help="Store /hits in output file")
    parser.add_argument("--store_clusters", action='store_true', help="Store /clusters in output file")
    parser.add_argument("--store_events", action='store_true', help="Store /events in output file")

    # Time options
    # parser.add_argument("-e", "--exposure", type=int, help='Exposure time (default: inf)')
    # parser.add_argument("-f", "--frame, type=int, help='Which frame to show')
    parser.add_argument("--spidr_stats", action='store_true', help='Print SPIDR timer stats (default: false)')

    # Frame options
    parser.add_argument("--frame_hits", action='store_true', help='Show counting mode frame of hits (default: false)')
    parser.add_argument("--frame_clusters", action='store_true', help='Show clusters (default: false)')
    parser.add_argument("--show", action='store_true', help='Show frame in window (default: false)')

    # Statistics
    parser.add_argument("--stats", action='store_true', help='Print image stats (default: false)')

    # Constants
    parser.add_argument("--cores", type=int, help='Number of cores to use')
    parser.add_argument("-a", "--algorithm", help='Event localisation algorithm to use')


    args = parser.parse_args(remaining_argv)

    return args

