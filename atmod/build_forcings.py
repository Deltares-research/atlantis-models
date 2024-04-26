import numpy as np
import xarray as xr

from atmod.base import Raster, VoxelModel


def surcharge_like(
    other: Raster | VoxelModel | xr.Dataset | xr.DataArray,
    lithology: int | np.ndarray,
    thickness: float | np.ndarray,
    times: np.datetime64 | np.ndarray,
):
    """
    Create an Xarray Dataset for the Atlantis surcharge forcing based on the y,x extent
    of an input model like object. The surcharge dataset that is created contains the
    lithological profile with corresponding thicknesses in each y,x cell for the complete
    dataset.

    Parameters
    ----------
    other : Raster | VoxelModel | xr.Dataset | xr.DataArray
        Input to base the y and x dimensions on for the surcharge dataset to create.
    lithology : int | np.array
        Lithological profile or single cell to add as a surcharge.
    thickness : float | np.array
        Corresponding thickness to each lithology.
    times : np.datetime64 | np.array
        Times at which the surcharge must be added.

    Returns
    -------
    xr.Dataset
        Surcharge dataset.

    """  # noqa: E501
    xshape, yshape = len(other['x']), len(other['y'])

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
        y=(['y'], other['y'].values),
        x=(['x'], other['x'].values),
    )

    return xr.Dataset(data, coords)


def stage_indexation_from(
    weir_areas: xr.DataArray,
    like: Raster | VoxelModel | xr.DataArray | xr.Dataset,
    times: np.datetime64 | np.ndarray,
    factor: float | np.ndarray,
):
    """
    Create an Xarray Dataset for the Atlantis stage indexation forcing for an input
    DataArray of weir areas based on the y,x extent of an input model like object and
    for each time the stage indexation forcing must be applied.

    Parameters
    ----------
    weir_areas : xr.DataArray
        2D DataArray with dims y,x containing weir areas over which stage indexation is
        calculated.
    like : Raster | VoxelModel | xr.DataArray | xr.Dataset
        Input to base the y and x dimensions on for the stage indexation dataset to
        create.
    times : np.datetime64 | np.ndarray
        Times at which the stage indexation must be applied in the modelling.
    factor : float | np.ndarray
        Factor of the subsidence by which stage indexation is calculated.

    Returns
    -------
    xr.Dataset
        Stage indexation dataset.

    """
    weir_areas = weir_areas.sel(x=like['x'], y=like['y'], method='nearest')

    times, ntimes = _get_dim_input(times.astype('datetime64[ns]'))
    weir_areas = repeat_in_time(weir_areas, ntimes)

    if isinstance(factor, (int, float)):
        factor = np.full_like(weir_areas, factor, dtype='float64')
    else:
        factor = repeat_in_time(factor, ntimes)

    data = dict(
        weir_area=(['time', 'y', 'x'], weir_areas), factor=(['time', 'y', 'x'], factor)
    )
    coords = dict(time=times, y=like['y'], x=like['x'])

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


def repeat_in_time(arr, times):
    """
    Repeat a 2D array n times along dimension time (axis=0).

    """
    return np.tile(arr, (times, 1, 1))
