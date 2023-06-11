# We test for this already in tpx3hitparser.py and print warnings for which version is loaded
try:
    try:
        from tpx3hitparser.clusters.clfind.target.release.libclfind import clfind
    except ModuleNotFoundError:
        from tpx3hitparser.clusters.clfind.target.debug.libclfind import clfind
except ImportError:
    from tpx3hitparser.clusters.clfind.clfind_np import clfind