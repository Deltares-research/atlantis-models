import rasterio
import numpy as np
import rioxarray as rio
import xarray as xr
from abc import ABC, abstractmethod
from pathlib import WindowsPath
from typing import Union
from rasterio.crs import CRS
from rasterio.enums import Resampling


class Spatial(ABC):
    """
    Abstract base class for spatial objects.

    """

    @property
    @abstractmethod
    def bounds(self):
        pass

    @property
    @abstractmethod
    def xmin(self):
        pass

    @property
    @abstractmethod
    def xmax(self):
        pass

    @property
    @abstractmethod
    def ymin(self):
        pass

    @property
    @abstractmethod
    def ymax(self):
        pass

    @property
    @abstractmethod
    def crs(self):
        pass


class Raster(Spatial):
    def __init__(
        self,
        ds: xr.DataArray,
        cellsize: Union[int, float],
        crs: Union[str, int, CRS] = None,
    ):
        self.ds = ds
        self.cellsize = cellsize

        if crs is not None:
            self.ds.rio.write_crs(crs)

    @classmethod
    def from_tif(cls, tif_path, bbox=None):
        ds = rio.open_rasterio(tif_path)
        ds = ds.sel(band=1)
        xres, _ = ds.rio.resolution()
        crs = ds.rio.crs

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

        return cls(ds, xres, crs)

    @property
    def xcoords(self):
        return self.ds['x'].values

    @property
    def ycoords(self):
        return self.ds['y'].values

    @property
    def coords(self) -> dict:
        return self.ds.coords

    @property
    def dims(self) -> tuple:
        return self.ds.dims

    @property
    def nrows(self) -> int:
        return len(self.ycoords)

    @property
    def ncols(self) -> int:
        return len(self.xcoords)

    @property
    def shape(self):
        return self.nrows, self.ncols

    @property
    def xmin(self):
        xmin = np.min(self.xcoords)
        return xmin - (0.5 * self.cellsize)

    @property
    def xmax(self):
        xmax = np.max(self.xcoords)
        return xmax + (0.5 * self.cellsize)

    @property
    def ymin(self):
        ymin = np.min(self.ycoords)
        return ymin - (0.5 * self.cellsize)

    @property
    def ymax(self):
        ymax = np.max(self.ycoords)
        return ymax + (0.5 * self.cellsize)

    @property
    def bounds(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def crs(self):
        return self.ds.rio.crs

    @property
    def x_ascending(self):
        return self.xmax > self.xmin

    @property
    def y_ascending(self):
        return self.ymax > self.ymin

    @property
    def dtype(self):
        return self.ds.dtype

    def get_affine(self) -> tuple:
        """
        Get an affine matrix based on the 2D extent of the data.

        """
        x_rotation, y_rotation = 0.0, 0.0
        return self.ncols, x_rotation, self.xmin, y_rotation, -self.nrows, self.ymax

    def set_cellsize(
        self,
        cellsize: Union[int, float],
        resampling_method=Resampling.bilinear,
        inplace=False,
    ):
        upscale_factor =  self.cellsize / cellsize
        new_width = int(self.ncols * upscale_factor)
        new_height = int(self.nrows * upscale_factor)

        ds_resampled = self.ds.rio.reproject(
            self.crs, shape=(new_height, new_width), resampling=resampling_method
        )
        if inplace:
            self.ds = ds_resampled
            self.cellsize = cellsize
        else:
            return self.__class__(ds_resampled, cellsize, self.crs)

    def set_crs(self, crs: Union[str, int, CRS]):
        self.ds.rio.write_crs(crs, inplace=True)

    def to_tiff(self, outputpath):
        pass


class VoxelModel(Raster):
    def __init__(
        self,
        ds: xr.Dataset,
        cellsize: Union[int, float],
        dz: Union[int, float],
        crs: Union[str, int, CRS] = None,
    ):
        Raster.__init__(self, ds, cellsize)
        self.dz = dz

    def __repr__(self):
        instance = f'atmod.{self.__class__.__name__} instance'
        layers = self.ds.data_vars
        dimensions = f'Dimensions: {dict(self.dims)}'
        resolution = f'Resolution (y, x, z): {self.cellsize, self.cellsize, self.dz}'
        return f'{instance}\n{layers}\n{dimensions}\n{resolution}'

    def __getitem__(self, item):
        return self.ds[item]

    def __setitem__(self, key, item):
        self.ds[key] = item

    @classmethod
    def from_tif(cls, *_):
        raise NotImplementedError("Cannot create VoxelModel from tif.")

    @property
    def zcoords(self):
        return self.ds['z'].values

    @property
    def zmin(self):
        zmin = np.min(self.zcoords)
        return zmin - (0.5 * self.dz)

    @property
    def zmax(self):
        zmax = np.max(self.zcoords)
        return zmax + (0.5 * self.dz)

    @property
    def nz(self):
        return len(self.zcoords)

    @property
    def shape(self):
        return self.nrows, self.ncols, self.nz

    @property
    def data_vars(self):
        return list(self.ds.data_vars.keys())

    @property
    def dtype(self):
        return self.ds.dtype

    def select_like(self, other):
        other_y = other.ycoords
        other_x = other.xcoords
        other_z = other.zcoords

        sel = self.ds.sel(y=other_y, x=other_x, z=other_z, method='nearest')
        sel = sel.assign_coords({'y': other_y, 'x': other_x, 'z': other_z})
        return self.__class__(sel, other.cellsize, other.dz)

    def drop_vars(self, data_vars: Union[str, list], inplace=True):
        if inplace:
            self.ds = self.ds.drop_vars(data_vars)
        else:
            return self.__class__(self.ds.drop_vars(data_vars), self.cellsize, self.dz)

    def zslice_to_tiff(
        self, layer: str, z: Union[int, float], outputpath: Union[str, WindowsPath]
    ):
        zslice = self.ds[layer].sel(z=z)

        if not self.x_ascending:
            zslice = zslice.isel(x=slice(None, None, -1))
        if self.y_ascending:
            zslice = zslice.isel(y=slice(None, None, -1))

        affine = self.get_affine()
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': self.ncols,
            'height': self.nrows,
            'crs': self.crs.to_string(),
            'nodata': np.nan,
            'transform': affine,
            'count': 1,
        }
        with rasterio.open(outputpath, 'w', **meta) as dst:
            dst.write(zslice.values, 1)

    @property
    def isvalid_area(self):
        """
        2D DataArray where GeoTop contains valid voxels at y, x locations.
        """
        if not hasattr(self, '_isvalid'):
            self.get_isvalid_area()
        return self._isvalid_area

    @property
    def ismissing_area(self):
        """
        2D DataArray where GeoTop has no data at y, x locations.
        """
        return ~self._isvalid_area

    @property
    def isvalid(self):
        return ~np.isnan(self.ds[self.data_vars[0]])

    def get_isvalid_area(self):
        self._isvalid_area = np.any(self.isvalid, axis=2)
        return self._isvalid_area

    def mask_surface_level(self):
        max_idx_valid = np.argmax(np.cumsum(self.isvalid.values, axis=2), axis=2)
        return max_idx_valid


class Mapping(Spatial):
    def __init__(self, gdf):
        self.gdf = gdf

    @property
    def bounds(self):
        return self.gdf.total_bounds

    @property
    def xmin(self):
        return self.bounds[0]

    @property
    def xmax(self):
        return self.bounds[2]

    @property
    def ymin(self):
        return self.bounds[1]

    @property
    def ymax(self):
        return self.bounds[3]

    @property
    def crs(self):
        self.gdf.crs
