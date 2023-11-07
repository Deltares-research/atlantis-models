import sqlite3
import numpy as np
import xarray as xr
from collections import OrderedDict
from pathlib import WindowsPath
from sqlite3 import Error
from typing import Union

from atmod.base import Raster, VoxelModel


def create_connection(database: Union[str, WindowsPath]):
    """
    Create a database connection to an SQLite database.

    Parameters
    ----------
    database: string
        Path/url/etc. to the database to create the connection to.

    Returns
    -------
    conn : sqlite3.Connection
        Connection object or None.

    """
    conn = None
    try:
        conn = sqlite3.connect(database)
    except Error as e:
        print(e)

    return conn


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


def get_xcoordinates(xllcenter: Union[int, float], ncols: int, cellsize: int):
    xmin = xllcenter
    xmax = xmin + (ncols * cellsize)
    return np.arange(xmin, xmax, cellsize)


def get_ycoordinates(yllcenter: Union[int, float], nrows: int, cellsize: int):
    ymin = yllcenter - cellsize  # subtract cellsize to include in descending np.arange
    ymax = ymin + (nrows * cellsize)
    return np.arange(ymax, ymin, -cellsize)


def get_zcoordinates(
    zmin: Union[int, float], zmax: Union[int, float], dz: Union[int, float]
):
    return np.arange(zmin, zmax + dz, dz)


def _follow_gdal_conventions(ds):
    if "z" in ds.dims:
        ds = ds.transpose("y", "x", "z")
    else:
        ds = ds.transpose("y", "x")
    return ds


def get_full_like(
    model: Union[Raster, VoxelModel],
    fill_value: float,
    invalid_value: float = None,
    dtype="float32",
):
    result = np.full(model.shape, fill_value, dtype=dtype)
    result[~model.isvalid] = invalid_value
    return result
