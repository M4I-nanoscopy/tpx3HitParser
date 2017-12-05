import ConfigParser
import argparse
import logging
import sys
import os

logger = logging.getLogger('root')

settings = None


def parse_config(argv=None):
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
    conf_parser.add_argument("-c", "--config", help="Specify config file for constant defaults (default: default.cfg)",
                             metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    default_conf_file = os.path.join(os.path.dirname(argv[0]), 'default.cfg')

    conf_file = None
    if not args.config and os.path.exists(default_conf_file):
        conf_file = default_conf_file
    elif args.config and os.path.exists(args.config):
        conf_file = args.config
    elif args.config and os.path.exists(args.config):
        logger.error('Config file %s does not exist' % args.config)
        raise Exception

    # Read defaults from config file
    defaults = {}
    if conf_file:
        config_parser = ConfigParser.SafeConfigParser()
        config_parser.read([conf_file])
        defaults = dict(config_parser.items("Defaults"))
    else:
        logger.warn("No config file being used for setting constant defaults")

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35, width=100)
    )
    parser.set_defaults(**defaults)

    # Input arguments
    input_group = parser.add_argument_group('input arguments')
    input_group_excl = input_group.add_mutually_exclusive_group(required=True)
    input_group_excl.add_argument("--raw", metavar='FILE', help="Read raw .tpx3")
    input_group_excl.add_argument("--hits", metavar='FILE', help="Read .h5 file containing /hits")
    input_group_excl.add_argument("--clusters", metavar='FILE', help="Read .h5 file containing /clusters")

    # Parse options
    parse_group = parser.add_argument_group('parse arguments')
    parse_group.add_argument("-C", action='store_true', help="Parse clusters")
    parse_group.add_argument("-E", action='store_true', help="Parse events")

    # Output file arguments
    output_group = parser.add_argument_group('output arguments')
    output_group.add_argument("-o", "--output", metavar='FILE', help='Output HDF5 file')
    output_group.add_argument("--store_hits", action='store_true', help="Store /hits in output file")
    output_group.add_argument("--store_clusters", action='store_true', help="Store /clusters in output file")
    output_group.add_argument("--store_events", action='store_true', help="Store /events in output file")

    # Misc options
    misc_group = parser.add_argument_group('miscellaneous arguments')
    misc_group.add_argument("--spidr_stats", action='store_true', help='Print SPIDR timer stats')
    misc_group.add_argument("--frame_hits", action='store_true', help='Show counting mode frame of hits')
    misc_group.add_argument("--stats", action='store_true', help='Print hit frame stats')

    # Constants
    constants_group = parser.add_argument_group('constants')
    constants_group.add_argument("--cores", type=int, help='Number of cores to use (default: %s) ' % defaults['cores'])
    constants_group.add_argument("-a", "--algorithm", metavar='A',
                                 help='Event localisation algorithm to use (default: %s)' % defaults['algorithm'])
    constants_group.add_argument("--cluster_min_size", metavar='N',
                                 help='Minimum cluster size (default: %s)' % defaults['cluster_min_size'])
    constants_group.add_argument("--cluster_max_size", metavar='N',
                                 help='Maximum cluster size (default: %s)' % defaults['cluster_max_size'])
    constants_group.add_argument("--cluster_max_sum_tot", metavar='N',
                                 help='Maximum cluster sum tot (default: %s)' % defaults['cluster_max_sum_tot'])
    constants_group.add_argument("--cluster_min_sum_tot", metavar='N',
                                 help='Minimum cluster sum tot (default: %s)' % defaults['cluster_min_sum_tot'])
    constants_group.add_argument("--cluster_chunk_size", type=int, metavar='N',
                                 help='Number of hits to consider at once (memory intensive!) (default: %s) ' % defaults['cluster_chunk_size'])
    constants_group.add_argument("--cluster_matrix_size", type=int, metavar='N',
                                 help='Size of the resulting cluster matrix (default: %s) ' % defaults['cluster_matrix_size'])

    # Misc
    parser.add_argument("-v", "--verbose", action='store_true', help='Verbose output')

    global settings
    settings = parser.parse_args(remaining_argv)

    if (settings.store_hits or settings.store_clusters or settings.store_events) and not settings.output:
        parser.error('An output file (-o or --output) is required when specifying a store option')
