import logging
logger = logging.getLogger('root')

try:
    try:
        from clusters.clfind.target.release.libclfind import clfind
    except ModuleNotFoundError:
        from clusters.clfind.target.debug.libclfind import clfind
        logger.warning("WARNING: Loaded debug version of Rust compiled clfind (this is slower).")
except ImportError:
    logger.warning("WARNING: Could not find or load the compiled Rust version of clfind. Loading slower numpy implementation")
    from clusters.clfind.clfind_np import clfind