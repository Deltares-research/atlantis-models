import numpy as np
import xarray as xr

from atmod.base import Raster, VoxelModel


def surcharge_like(
    ds: Raster | VoxelModel | xr.Dataset | xr.DataArray,
    lithology: int | np.ndarray,
    thickness: float | np.ndarray,
    times: np.datetime64 | np.ndarray,
):
    """
    _summary_

    Parameters
    ----------
    ds : Raster | VoxelModel | xr.Dataset | xr.DataArray
        _description_
    lithology : int | np.array
        _description_
    thickness : float | np.array
        _description_
    times : np.datetime64 | np.array
        _description_

    Returns
    -------
    _type_
        _description_
    """
    xshape, yshape = len(ds['x']), len(ds['y'])

    times, ntimes = _get_dim_input(times.astype('datetime64[ns]'))
    lithology, nlayers = _get_dim_input(lithology)
    thickness, _nlayers = _get_dim_input(thickness)

    if nlayers != _nlayers:
        raise ValueError('lithology and thickness must have the same number of layers')

    layers = np.arange(nlayers) + 1

    lithology = _repeat(lithology, ntimes, nlayers, yshape, xshape)
    thickness = _repeat(thickness, ntimes, nlayers, yshape, xshape)

    data = dict(
        lithology=(['time', 'layer', 'y', 'x'], lithology),
        thickness=(['time', 'layer', 'y', 'x'], thickness),
    )

    coords = dict(
        time=(['time'], times),
        layer=(['layer'], layers),
        y=(['y'], ds['y'].values),
        x=(['x'], ds['x'].values),
    )

    return xr.Dataset(data, coords)


def _get_dim_input(variable):
    """
    Helper function for surcharge_like to get correct input for the Dataset coordinates
    in the surcharge to build.

    """
    try:
        length = len(variable)
    except TypeError:
        length = 1
        variable = [variable]
    return variable, length


def _repeat(array: np.ndarray, ntimes: int, nlayers: int, ny: int, nx: int):
    """
    Helper function to repeat lithological or thickness layers n times along
    dimensions time, y and x with a specific ordering of dims time, layer, y, x.

    Parameters
    ----------
    array : np.ndarray
        Numpy array of the lithological or thickness to repeat.
    ntimes : int
        N repeats along dimension time.
    nlayers : int
        Number of layers present in the lithology or thickness array.
    ny : int
        N repeats along dimension y.
    nx : int
        N repeats along dimension x.

    Returns
    -------
    np.ndarray
        Numpy array with the repeats with dimensions (time, layer, y, x).
    """
    if nlayers == 1:
        array = np.full((ntimes, nlayers, ny, nx), array)
    else:
        array = np.tile(
            array[np.newaxis, :, np.newaxis, np.newaxis], (ntimes, 1, ny, nx)
        )
    return array
