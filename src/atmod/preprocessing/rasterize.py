from pathlib import WindowsPath

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features

from atmod.base import Raster, VoxelModel


def soilmap_to_raster(soilmap, da) -> Raster:
    gdf = soilmap.gdf
    gdf["nr"] = gdf["maparea_id"].str.split(".", expand=True)[2].astype(int)
    soilmap_da = rasterize_like(gdf, "nr", da)
    return Raster(soilmap_da, da.cellsize)


def rasterize_like(
    shapefile: str | WindowsPath | gpd.GeoDataFrame,
    attribute: str,
    da: Raster | VoxelModel,
):
    """
    Rasterize a shapefile like an atmod Raster or into the 2D extent of a VoxelModel
    object.

    Parameters
    ----------
    shapefile : str | WindowsPath | gpd.GeoDataFrame
        Input shapefile to rasterize. Can be a path to the shapefile or an in
        memory GeoDataFrame.
    attribute : str
        Name of the attribute in the shapefile to rasterize.
    da : Raster | VoxelModel,
        Atmod Raster or VoxelModel object to rasterize the shapefile like.
    cellsize : int, optional
        Cellsize of the output DataArray. The default is None, then the x and y
        size will be derived from the input DataArray.

    Returns
    -------
    xr.DataArray
        DataArray of the rasterized shapefile.

    """
    if isinstance(shapefile, (str, WindowsPath)):
        shapefile = gpd.read_file(shapefile)

    shapes = ((geom, z) for z, geom in zip(shapefile[attribute], shapefile["geometry"]))

    rasterized = features.rasterize(
        shapes=shapes,
        fill=np.nan,
        out_shape=(da.nrows, da.ncols),
        transform=da.get_affine(),
    )
    rasterized = xr.DataArray(rasterized, coords=da.coords, dims=da.dims)
    return rasterized


if __name__ == "__main__":
    # TODO: Move code below to tests
    from atmod.bro_models import BroBodemKaart
    from atmod.templates import build_template

    path_gpkg = r"c:\Users\knaake\OneDrive - Stichting Deltares\Documents\data\dino\bro_bodemkaart.gpkg"  # noqa: E501
    soilmap = BroBodemKaart.from_geopackage(path_gpkg)

    print("Build template")
    da = build_template(50, 50, 122050, 446050, 100)
    xmin, xmax = 120_000, 130_000
    ymin, ymax = 440_000, 455_000

    print("Read soilmap")
    map_ = soilmap.read_soilmap(bbox=(xmin, ymin, xmax, ymax))

    map_["nr"] = map_["maparea_id"].str.split(".", expand=True)[2].astype(int)

    print("Rasterize")
    test = rasterize_like(map_, "nr", da)

    print(2)
