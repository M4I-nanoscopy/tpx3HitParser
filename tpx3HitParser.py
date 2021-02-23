#!/usr/bin/env python
import os
import sys
import lib
import logging
import orchestration
import tpx3format
import numpy

MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)

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

    # Check input
    if not os.path.exists(settings.raw):
        logger.error("Input file %s does not exist." % settings.raw)
        return 1

    # Basic output check (output writing can still fail on other factors (like no permissions)
    c = io.check_write(settings.output, settings.overwrite)
    if c is not True:
        logger.error(c)
        return 1

    # Check for CNN model file
    if settings.E and settings.algorithm == 'cnn' and not os.path.exists(settings.event_cnn_model):
        logger.error('CNN model %s does not exist.' % settings.event_cnn_model)
        return 1

    # Check if we have a loadable ToT correction file
    c = tpx3format.check_tot_correction(settings.hits_tot_correct_file)
    if c is not True:
        logger.error(c)
        return 1

    # Test the importing of the compiled Rust library of clfind
    # This is a copy of the code in clusters/clfind/__init__.py
    try:
        try:
            from clusters.clfind.target.release.libclfind import clfind
        except ModuleNotFoundError:
            from clusters.clfind.target.debug.libclfind import clfind
            logger.warning("Loaded debug version of Rust compiled clfind (this is slower).")
    except ImportError:
        logger.warning(
            "Could not find or load the compiled Rust version of clfind. Loading slower numpy implementation")
        from clusters.clfind.clfind_np import clfind

    # Start main processing
    orchestrator = orchestration.Orchestrator(settings)
    try:
        orchestrator.orchestrate()
    finally:
        # We hit an exception or are done, terminate workers and progress bar
        orchestrator.cleanup()

    # Post parsing correction utilities ###

    # ToA sorting for hits and events
    if settings.hits_sort_toa:
        logger.info('Start sorting hits data on ToA...')
        hits = io.read_hits(settings.output)
        hits = numpy.sort(hits, 0, 'stable', 'ToA')
        io.open_write(settings.output, overwrite=False, append=True)
        io.replace_hits(hits)
        io.close_write()
        logger.info('Finished sorting hits data on ToA.')

    if settings.event_sort_toa:
        logger.info('Start sorting event data on ToA...')
        events = io.read_events(settings.output)
        events = numpy.sort(events, 0, 'stable', 'ToA')
        io.open_write(settings.output, overwrite=False, append=True)
        io.replace_events(events)
        io.close_write()
        logger.info('Finished sorting event data on ToA.')

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
