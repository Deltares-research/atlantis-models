import numba
import warnings


class IgnoreNumbaTypeSafetyWarning:
    """
    Context manager class to ignore Numba type safety warnings.
    """
    def __enter__(self):
        warnings.filterwarnings("ignore", category=numba.NumbaTypeSafetyWarning)
        return self

    def __exit__(self, *args):
        warnings.resetwarnings()


class IgnoreRuntimeWarning:
    """
    Context manager class to ignore RuntimWarnings for e.g. zero division warnings.
    """
    def __enter__(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *args):
        warnings.resetwarnings()
