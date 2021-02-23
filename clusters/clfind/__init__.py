# We test for this already in tpx3HitParser.py and print warnings for which version is loaded
try:
    try:
        from clusters.clfind.target.release.libclfind import clfind
    except ModuleNotFoundError:
        from clusters.clfind.target.debug.libclfind import clfind
except ImportError:
    from clusters.clfind.clfind_np import clfind