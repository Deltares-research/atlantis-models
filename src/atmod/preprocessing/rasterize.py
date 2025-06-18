from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features

from atmod.base import VoxelModel


def soilmap_to_raster(soilmap, da):
    gdf = soilmap.gdf
    gdf["nr"] = gdf["maparea_id"].str.split(".", expand=True)[2].astype(int)
    return rasterize_like(gdf, da, "nr", fill=np.iinfo(np.int64).min)


def rasterize_like(
    shapefile: str | Path | gpd.GeoDataFrame,
    da: xr.DataArray,
    attribute: str = None,
    **features_kwargs,
):
    """
    Rasterize a shapefile like an Xarray DataArray object.

    Parameters
    ----------
    shapefile : str | Path | gpd.GeoDataFrame
        Input shapefile to rasterize. Can be a path to the shapefile or an in
        memory GeoDataFrame.
    da : xr.DataArray,
        DataArray to use the extent from rasterize the shapefile like.
    attribute : str, optional
        Name of the attribute in the shapefile to rasterize. The default is None, in
        this case, a default value of 1 will be burnt into the DataArray.

    **features_kwargs
        See rasterio.features.rasterize docs for additional optional parameters.

    Returns
    -------
    xr.DataArray
        DataArray of the rasterized shapefile.

    Examples
    --------
    Rasterize a specific attribute of a shapefile:
    >>> rasterize_like(shapefile, da, "attribute")

    Use additional `features.rasterize` options:
    >>> rasterize_like(shapefile, da, "attribute", fill=np.nan, all_touched=True)

    """
    if isinstance(da, VoxelModel):
        da = da.ds

    if isinstance(shapefile, (str, Path)):
        shapefile = gpd.read_file(shapefile)

    if attribute:
        shapes = (
            (geom, z) for z, geom in zip(shapefile[attribute], shapefile["geometry"])
        )
    else:
        shapes = (geom for geom in shapefile["geometry"])
        features_kwargs["default_value"] = 1

    rasterized = features.rasterize(
        shapes=shapes,
        out_shape=da.shape,
        transform=da.rio.transform(),
        **features_kwargs,
    )

    return xr.DataArray(rasterized, coords=da.coords, dims=da.dims)
