import warnings
from functools import wraps


class WarningSuppressor:
    def __init__(self, *warning_types):
        self.warning_types = warning_types

    def __enter__(self):
        self._original_showwarning = warnings.showwarning
        warnings.simplefilter("ignore", *self.warning_types)

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.showwarning = self._original_showwarning


def suppress_warnings(*warning_types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with WarningSuppressor(*warning_types):
                return func(*args, **kwargs)

        return wrapper

    return decorator
