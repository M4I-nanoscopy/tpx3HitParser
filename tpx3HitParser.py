#!/usr/bin/env python

import sys
import lib
import logging
import orchestration

logger = lib.setup_custom_logger('root', logging.INFO)


def main():
    # Parse all command line arguments
    lib.parse_config()
    settings = lib.config.settings

    if settings.verbose:
        log_level = logging.DEBUG
        logger.setLevel(log_level)

    io = lib.io()

    # Print config statements
    logger.debug(settings)

    # TODO: Check input file
    # TODO: Check CNN model
    # TODO: Check output file

    # Check if we have a loadable ToT correction file
    # tpx3format.check_tot_correction(lib.config.settings.hits_tot_correct_file)

    orchestrator = orchestration.Orchestrator(settings)

    try:
        orchestrator.orchestrate()
    finally:
        # We hit an exception or are done, terminate workers and progress bar
        orchestrator.cleanup()

    # Post event parsing corrections ###

    # # Super Resolution
    # if settings.correct_super_res:
    #     e = events.subpixel_event_redistribution(e)
    #
    # # Correct chip edges
    # if settings.correct_chip_edge:
    #     e = events.chip_edge_correct(e)
    #
    # if settings.store_predictions:
    #     io.store_predictions(events.calculate_predictions(e, cluster_info), settings.algorithm)

    # io.close_write()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except lib.UserConfigException as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
