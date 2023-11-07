import rioxarray as rio
from atmod.base import Raster
from atmod.utils import _follow_gdal_conventions


def read_ahn(tif_path, cellsize: int = 100, bbox: tuple = None):
    ds = rio.open_rasterio(tif_path)
    ds = ds.sel(band=1)

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    ahn = Raster(ds, cellsize)
    return ahn
