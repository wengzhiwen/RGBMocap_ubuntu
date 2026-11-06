"""Runtime facts that belong in inference receipts on a non-default Linux variant.

Windows and amd64 Linux get an empty addition and the unchanged PROFILES entry, so their
existing identities stay byte-identical. Only the standard library is used here.
"""
import os


def variant_identity():
    """Extra receipt keys; empty unless a separately pinned variant is in use."""
    if os.name=='nt':return {}
    from .linux_runtime import variant_identity as linux_identity
    return linux_identity()


def runtime_profile(key):
    """The runtime actually used by a profile, for receipts that embed it."""
    if os.name=='nt':
        from .runtime_registry import PROFILES
        return PROFILES[key]
    from .linux_runtime import effective_profile
    return effective_profile(key)
