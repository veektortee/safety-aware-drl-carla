"""Verbose per-step diagnostics are off unless the environment variable CARLA_DEBUG=1 is set."""
import os

DEBUG = os.environ.get("CARLA_DEBUG", "0") == "1"


def dprint(*args, **kwargs):
    """print() that only fires when CARLA_DEBUG=1."""
    if DEBUG:
        print(*args, **kwargs)
