"""Peak memory of the current process, in bytes, on every supported platform.

Only reported in stage events and receipts for the user; it never selects a path.
Importing this module loads no third-party package, so it cannot disturb the
thread limits workers configure before NumPy/Torch load.
"""
import os
import sys


def peak_rss_bytes():
    """Windows peak working set, or the POSIX peak resident set size."""
    if os.name=='nt':
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # getrusage reports KiB on Linux, but bytes on macOS.
    return peak if sys.platform=='darwin' else peak*1024
