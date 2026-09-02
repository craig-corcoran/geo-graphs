"""Stand-in for upstream apls_plots, so the vendored scorer needs no edits.

apls_reference.py calls plot_metric() to write diagnostic PNGs. Keeping that
call site intact is what lets the scorer stay byte-for-byte upstream, but a test
oracle has no business writing figures, so the call lands here and does nothing.
"""


def plot_metric(*args, **kwargs) -> None:
    """Accept and discard an upstream plotting request."""
