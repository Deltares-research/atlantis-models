import dask.array as darray
import numpy as np
import xarray as xr
from collections import OrderedDict
from typing import Union
from atmod.base import Raster, VoxelModel
from atmod.utils import get_xcoordinates, get_ycoordinates, get_zcoordinates


def dask_output_model_like(
    voxelmodel: VoxelModel, chunksize: int, add_phreatic=False, add_soilmap_layers=True
):
    ny, nx, nz = voxelmodel.shape

    out_zcoords = voxelmodel['z'].values

    if add_soilmap_layers:
        max_layers_soilmap = 9
        nz += max_layers_soilmap

        extra_zcoords = np.arange(0, voxelmodel.dz * max_layers_soilmap, voxelmodel.dz)
        extra_zcoords = extra_zcoords + np.max(out_zcoords) + voxelmodel.dz

        out_zcoords = np.append(out_zcoords, extra_zcoords)

    empty_3d = darray.empty(
        shape=(ny, nx, nz), dtype='float64', chunks=(chunksize, chunksize, nz)
    )
    empty_2d = darray.empty(
        shape=(ny, nx), dtype='float64', chunks=(chunksize, chunksize)
    )

    data = dict(
        geology=(['y', 'x', 'z'], empty_3d),
        lithology=(['y', 'x', 'z'], empty_3d),
        thickness=(['y', 'x', 'z'], empty_3d),
        mass_fraction_organic=(['y', 'x', 'z'], empty_3d),
        surface_level=(['y', 'x'], empty_2d),
        rho_bulk=(['y', 'x', 'z'], empty_3d),
        zbase=(['y', 'x'], empty_2d),
        max_oxidation_depth=(['y', 'x'], empty_2d),
        no_oxidation_thickness=(['y', 'x'], empty_2d),
        no_shrinkage_thickness=(['y', 'x'], empty_2d),
        domainbase=(['y', 'x'], empty_2d),
    )
    if add_phreatic:
        data['phreatic_level'] = (['y', 'x'], empty_2d)

    coords = dict(y=voxelmodel['y'], x=voxelmodel['x'], z=out_zcoords)

    return xr.Dataset(data, coords)


def build_template(
    ncols: int,
    nrows: int,
    xllcenter: Union[int, float],
    yllcenter: Union[int, float],
    cellsize: int,
    zmin: Union[int, float] = None,
    zmax: Union[int, float] = None,
    dz: Union[int, float] = 0.5,
):
    """
    Build a template [y, x, (z)] model in which to place data for Atlantis models.

    Parameters
    ----------
    ncols : int
        Number of columns in horizontal direction.
    nrows : int
        Number of rows in vertical direction.
    xllcenter : Union[int, float]
        Xcoordinate of cellcenter in the lower left corner.
    yllcenter : Union[int, float]
        Ycoordinate of cellcenter in the lower left corner.
    cellsize : int
        Cellsize [x, y] of the model.
    zmin : Union[int, float], optional
        Depth of the voxel mid of the lowest voxel for a 3D template, the default
        is None.
    zmax : Union[int, float], optional
        Depth of the voxel mid of the highest voxel for a 3D template, the default
        is None.
    dz : Union[int, float], optional
        Vertical thickness of each voxel, the default is 0.5.

    Returns
    -------
    xr.DataArray
        DataArray of the template model.

    Raises
    ------
    ValueError
        If not both 'zmin' and 'zmax' are specified when creating a 3D template
        model.

    """
    coords = OrderedDict()
    coords["y"] = get_ycoordinates(yllcenter, nrows, cellsize)
    coords["x"] = get_xcoordinates(xllcenter, ncols, cellsize)

    if zmin is not None and zmax is not None:
        z = get_zcoordinates(zmin, zmax, dz)
        coords["z"] = z

        array = np.full((nrows, ncols, len(z)), 0)
        dims = ("y", "x", "z")
        chunks = {"y": 200, "x": 200, "z": len(z)}
    elif zmin is not None or zmax is not None:
        raise ValueError("Specify both zmin and zmax for 3D template model.")
    else:
        array = np.full((nrows, ncols), 0)
        dims = ("y", "x")
        chunks = {"y": 200, "x": 200}

    template = xr.DataArray(array, coords=coords, dims=dims).chunk(chunks)

    if "z" in template.dims:
        return VoxelModel(template, cellsize, dz)
    else:
        return Raster(template, cellsize)


def get_full_like(
    model: Union[Raster, VoxelModel],
    fill_value: float,
    invalid_value: float = None,
    dtype="float32",
):
    result = np.full(model.shape, fill_value, dtype=dtype)
    result[~model.isvalid] = invalid_value
    return result
