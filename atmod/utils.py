import sqlite3
import numpy as np
from pathlib import WindowsPath
from sqlite3 import Error
from typing import Union
from rasterio.crs import CRS


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


def get_crs_object(crs: Union[str, int, CRS]):
    if isinstance(crs, str):
        crs = CRS.from_string(crs)
    elif isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    elif isinstance(crs, CRS):
        crs = crs
    else:
        raise ValueError('Input crs not understood.')
    return crs
