# We test for this already in tp3hitparser.py and print warnings for which version is loaded
try:
    try:
        from tp3hitparser.clusters.clfind.target.release.libclfind import clfind
    except ModuleNotFoundError:
        from tp3hitparser.clusters.clfind.target.debug.libclfind import clfind
except ImportError:
    from tp3hitparser.clusters.clfind.clfind_np import clfind