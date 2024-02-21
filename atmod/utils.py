import sqlite3
import numpy as np
import xarray as xr
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

    if ds['y'][-1] > ds['y'][0]:
        ds = ds.sel(y=slice(None, None, -1))

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


def _interpolate_point(line, loc):
    """
    Return the location (i.e. distance) and x and y coordinates of an interpolated
    point along a Shapely LineString object.

    Parameters
    ----------
    line : LineString
        shapely.geometry.LineString object.
    loc : int, float
        Distance along the LineString to interpolate the point at.

    """
    p = line.interpolate(loc)
    return loc, p.x, p.y


def sample_along_line(ds, line, dist=None, nsamples=None, cut_edges=True):
    """
    Sample x and y dims of an Xarray Dataset or DataArray over distance along a
    Shapely LineString object. Sampling can be done using a specified distance
    or a specified number of samples that need to be taken along the line.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to sample. Must contain dimensions 'x' and 'y' that
        refer to the coordinates.
    line : LineString
        shapely.geometry.LineString object to use for sampling.
    dist : int, float, optional
        Distance between each sample along the line. Takes equally distant samples
        from the start of the line untill it reaches the end. The default is None.
    nsamples : int, optional
        Number of samples to take along the line between the beginning and the end.
        The default is None.

    Raises
    ------
    ValueError
        If both or none of dist and nsamples are specified.

    Returns
    -------
    ds_sel : xr.Dataset or xr.DataArray
        Sampled Dataset or DataArray with dimension 'dist' for distance.

    """
    if dist and nsamples:
        raise ValueError(
            "Cannot use 'dist' and 'nsamples' together, use one option."
            )

    elif dist:
        sample_locs = np.arange(0, line.length, dist)

    elif nsamples:
        sample_locs = np.linspace(0, line.length, nsamples)

    else:
        raise ValueError(
            "'dist' or 'nsamples' not specified, use one option."
            )

    samplepoints = np.array([_interpolate_point(line, loc) for loc in sample_locs])
    dist, x, y = samplepoints[:, 0], samplepoints[:, 1], samplepoints[:, 2]

    ds_sel = ds.sel(
        x=xr.DataArray(x, dims='dist'),
        y=xr.DataArray(y, dims='dist'),
        method='nearest'
    )
    ds_sel = ds_sel.assign_coords(dist=('dist', dist))

    return ds_sel.transpose('z', 'dist')
