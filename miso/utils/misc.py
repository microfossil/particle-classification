"""
Miscellaneous functions
"""


def wrap_fn(fn, *args, **kwargs):
    """
    Wraps a function into one that takes a single input
    """
    def wrapper(x):
        return fn(x, *args, **kwargs)
    return wrapper