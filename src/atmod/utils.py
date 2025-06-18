import functools
import sqlite3
from pathlib import Path
from typing import TypeVar

import numpy as np
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling

VoxelModel = TypeVar("VoxelModel")
LineString = TypeVar("LineString")


COMPRESSION = {
    "geology": {"zlib": True, "complevel": 9},
    "lithology": {"zlib": True, "complevel": 9},
    "thickness": {"zlib": True, "complevel": 9},
    "mass_fraction_organic": {"zlib": True, "complevel": 9},
    "surface_level": {"zlib": True, "complevel": 9},
    "phreatic_level": {"zlib": True, "complevel": 9},
    "rho_bulk": {"zlib": True, "complevel": 9},
    "zbase": {"zlib": True, "complevel": 9},
    "max_oxidation_depth": {"zlib": True, "complevel": 9},
    "no_oxidation_thickness": {"zlib": True, "complevel": 9},
    "no_shrinkage_thickness": {"zlib": True, "complevel": 9},
    "domainbase": {"zlib": True, "complevel": 9},
}


def create_connection(database: str | Path):
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
    except sqlite3.Error as e:
        print(e)

    return conn


def get_xcoordinates(xllcenter: int | float, ncols: int, cellsize: int):
    xmin = xllcenter
    xmax = xmin + (ncols * cellsize)
    return np.arange(xmin, xmax, cellsize)


def get_ycoordinates(yllcenter: int | float, nrows: int, cellsize: int):
    ymin = yllcenter - cellsize  # subtract cellsize to include in descending np.arange
    ymax = ymin + (nrows * cellsize)
    return np.arange(ymax, ymin, -cellsize)


def get_zcoordinates(zmin: int | float, zmax: int | float, dz: int | float):
    return np.arange(zmin, zmax + dz, dz)


def _follow_gdal_conventions(ds):
    if "z" in ds.dims:
        ds = ds.transpose("y", "x", "z")
    else:
        ds = ds.transpose("y", "x")

    if ds["y"][-1] > ds["y"][0]:
        ds = ds.sel(y=slice(None, None, -1))

    return ds


def get_crs_object(crs: str | int | CRS):
    if isinstance(crs, str):
        crs = CRS.from_string(crs)
    elif isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    elif isinstance(crs, CRS):
        crs = crs
    else:
        raise ValueError("Input crs not understood.")
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


def sample_along_line(
    ds: xr.Dataset | xr.DataArray,
    line: LineString,
    dist: int | float = None,
    nsamples: int = None,
):
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
        raise ValueError("Cannot use 'dist' and 'nsamples' together, use one option.")

    elif dist:
        sample_locs = np.arange(0, line.length, dist)

    elif nsamples:
        sample_locs = np.linspace(0, line.length, nsamples)

    else:
        raise ValueError("'dist' or 'nsamples' not specified, use one option.")

    samplepoints = np.array([_interpolate_point(line, loc) for loc in sample_locs])
    dist, x, y = samplepoints[:, 0], samplepoints[:, 1], samplepoints[:, 2]

    ds_sel = ds.sel(
        x=xr.DataArray(x, dims="dist"), y=xr.DataArray(y, dims="dist"), method="nearest"
    )
    ds_sel = ds_sel.assign_coords(dist=("dist", dist))

    return ds_sel.transpose("z", "dist")


def divide_blocks(
    area: VoxelModel,
    ysize: int = None,
    xsize: int = None,
    real_units: bool = False,
):
    """
    Divide the area of a VoxelModel object into equal blocks of a specified
    'y' and 'x' size and get the bounding boxes of each block. Blocks are created
    starting from the top left corner of the area.

    Parameters
    ----------
    area : VoxelModel
        VoxelModel object to divide into blocks.
    ysize, xsize : int, optional
        Block size in y and x direction respectively. The default is None
    real_units : bool, optional
        If True, use real map units of the input area. If False, y and x sizes
        correspond with the number of cells in each direction. The default is False.

    Returns
    -------
    list
        List containing the bounding box tuples (xmin, ymin, xmax, ymax) for each block.

    """
    xmin, ymin, xmax, ymax = area.bounds

    if not ysize:
        ysize = ymax - ymin

    if not xsize:
        xsize = xmax - xmin

    if not real_units:
        ysize = ysize * area.cellsize
        xsize = xsize * area.cellsize

    block_bounds = []
    for ytop in np.arange(ymax, ymin, -ysize):
        ybottom = ytop - ysize
        if ybottom < ymin:
            ybottom = ymin

        for xleft in np.arange(xmin, xmax, xsize):
            xright = xleft + xsize
            if xright > xmax:
                xright = xmax

            block_bounds.append((xleft, ybottom, xright, ytop))

    return block_bounds


def find_overlapping_areas(ahn=None, geotop=None, nl3d=None, glg=None):
    bounds = []
    if ahn is not None:
        bounds.append(ahn.bounds)
    if geotop is not None:
        bounds.append(geotop.bounds)
    if nl3d is not None:
        bounds.append(nl3d.bounds)
    if glg is not None:
        bounds.append(glg.bounds)

    bounds = np.array(bounds)
    overlapping_bounds = (
        np.max(bounds[:, 0]),
        np.max(bounds[:, 1]),
        np.min(bounds[:, 2]),
        np.min(bounds[:, 3]),
    )
    return overlapping_bounds


def check_dims(func):
    @functools.wraps(func)
    def wrapper(ds):
        required_dims = {"y", "x", "z"}
        if not required_dims.issubset(ds.dims):
            raise ValueError(
                f"Dataset must contain dimensions {required_dims}, "
                f"but found {ds.dims}."
            )
        return func(ds)

    return wrapper


def set_cellsize(
    da: xr.DataArray,
    xsize: int | float,
    ysize: int | float,
    resampling_method=Resampling.bilinear,
) -> xr.DataArray:
    xres, yres = da.rio.resolution()
    new_width = abs(int(da.rio.width * (xres / xsize)))
    new_height = abs(int(da.rio.height * (yres / ysize)))

    resampled = da.rio.reproject(
        da.rio.crs, shape=(new_height, new_width), resampling=resampling_method
    )
    return resampled
