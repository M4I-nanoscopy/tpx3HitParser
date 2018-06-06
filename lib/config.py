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
    conf_parser.add_argument("-c", "--config", help="Specify other config file ",
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

        # Fix boolean values
        defaults['hits_remove_cross'] = config_parser.getboolean('Defaults', 'hits_remove_cross')
        defaults['hits_combine_chips'] = config_parser.getboolean('Defaults', 'hits_combine_chips')
        defaults['cluster_stats'] = config_parser.getboolean('Defaults', 'cluster_stats')
    else:
        logger.warning("No config file being used for setting constant defaults")

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35, width=100)
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
    output_group.add_argument("-o", "--output", required=True, metavar='FILE', help='Output HDF5 file')
    output_group.add_argument("--overwrite", action='store_true', help='Overwrite existing HDF5 file')
    output_group.add_argument("--store_hits", action='store_true', help="Store /hits in output file")
    output_group.add_argument("--store_clusters", action='store_true', help="Store /clusters in output file")
    output_group.add_argument("--store_events", action='store_true', help="Store /events in output file")

    # Misc options
    misc_group = parser.add_argument_group('miscellaneous arguments')
    misc_group.add_argument("--spidr_stats", action='store_true', help='Print SPIDR timer stats')
    misc_group.add_argument("--cluster_stats", action='store_true', help='Store cluster stats')
    misc_group.add_argument("--freq_tot", action='store_true', help="Parse and store ToT frequency matrix")

    # Constants
    c_group = parser.add_argument_group('constants')
    c_group.add_argument("--cores", metavar='N', type=int, help='Number of cores to use')
    c_group.add_argument("--max_hits", metavar='N', type=int, help='Maximum number of hits to read (0: infinite)')
    c_group.add_argument("--hits_remove_cross", metavar='0/1',  type=str2bool, help='Remove the middle border pixels between the chips')
    c_group.add_argument("--hits_combine_chips",metavar='0/1', type=str2bool, help='Combine the chips to one matrix')
    c_group.add_argument("--hits_cross_extra_offset", metavar='N', type=int, help='Extra offset used for the cross pixels per chip when combining the chips')
    c_group.add_argument("--hits_tot_correct_file", metavar='FILE', help='ToT correction file, or 0 for no correction')
    c_group.add_argument("--cluster_min_size", type=int, metavar='N', help='Minimum cluster size' )
    c_group.add_argument("--cluster_max_size", type=int, metavar='N', help='Maximum cluster size' )
    c_group.add_argument("--cluster_max_sum_tot", type=int, metavar='N', help='Maximum cluster sum tot' )
    c_group.add_argument("--cluster_min_sum_tot", type=int, metavar='N', help='Minimum cluster sum tot')
    c_group.add_argument("--cluster_chunk_size", type=int, metavar='N',
                                 help='Number of hits to consider at once (memory intensive!)')
    c_group.add_argument("--cluster_matrix_size", type=int, metavar='N', help='Size of the resulting cluster matrix')
    c_group.add_argument("--event_chunk_size", type=int, metavar='N', help='Number of events to process at once')
    c_group.add_argument("--event_cnn_model", metavar='FILE', help='CNN model to use for event localisation')
    c_group.add_argument("-a", "--algorithm", metavar='A', help='Event localisation algorithm to use')

    # Misc
    parser.add_argument("-v", "--verbose", action='store_true', help='Verbose output')

    global settings
    settings = parser.parse_args(remaining_argv)

    if settings.C and not (settings.raw or settings.hits):
        parser.error('Either --hits or --raw is required when parsing clusters (-C)')

    if settings.spidr_stats and not (settings.raw or settings.hits):
        parser.error('Either --hits or --raw is required when requesting --spidr_stats')


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')