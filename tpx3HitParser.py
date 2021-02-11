#!/usr/bin/env python
import os
import sys
import lib
import logging
import orchestration
import tpx3format

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
    # TODO: Check output file

    # Check for CNN model file
    if settings.E and settings.algorithm == 'cnn' and not os.path.exists(settings.event_cnn_model):
        logger.error('CNN model %s does not exist.' % settings.event_cnn_model)
        return 1

    # Check if we have a loadable ToT correction file
    c = tpx3format.check_tot_correction(settings.hits_tot_correct_file)
    if c is not True:
        logger.error(c)
        return 1

    orchestrator = orchestration.Orchestrator(settings)

    try:
        orchestrator.orchestrate()
    finally:
        # We hit an exception or are done, terminate workers and progress bar
        orchestrator.cleanup()

    # Post event parsing corrections ###
    # TODO: Reimplement these options
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

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except lib.UserConfigException as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
