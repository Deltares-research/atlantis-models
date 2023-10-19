import numpy as np
import xarray as xr
from abc import ABC, abstractmethod


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


class Raster(Spatial):
    def __init__(self, ds: xr.DataArray, cellsize: int):
        self.ds = ds
        self.cellsize = cellsize

    def __getitem__(self, item):
        return self.ds[item]

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
        return xmin - (0.5*self.cellsize)

    @property
    def xmax(self):
        xmax = np.max(self.xcoords)
        return xmax + (0.5*self.cellsize)

    @property
    def ymin(self):
        ymin = np.min(self.ycoords)
        return ymin - (0.5*self.cellsize)

    @property
    def ymax(self):
        ymax = np.max(self.ycoords)
        return ymax + (0.5*self.cellsize)

    @property
    def bounds(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_affine(self) -> tuple:
        """
        Get an affine matrix based on the 2D extent of the data.

        """
        x_rotation, y_rotation = 0.0, 0.0
        return self.ncols, x_rotation, self.xmin, y_rotation, -self.nrows, self.ymax


class VoxelModel(Raster):
    def __init__(self, ds, cellsize, dz):
        Raster.__init__(self, ds, cellsize)
        self.dz = dz

    @property
    def zcoords(self):
        return self.ds['z'].values

    @property
    def zmin(self):
        zmin = np.min(self.zcoords)
        return zmin - (0.5*self.dz)

    @property
    def zmax(self):
        zmax = np.max(self.zcoords)
        return zmax + (0.5*self.dz)

    @property
    def nz(self):
        return len(self.zcoords)

    @property
    def shape(self):
        return self.nrows, self.ncols, self.nz

    def select_like(self, other):
        other_y = other.ycoords
        other_x = other.xcoords
        other_z = other.zcoords

        sel = self.ds.sel(y=other_y, x=other_x, z=other_z, method='nearest')
        sel = sel.assign_coords({'y': other_y, 'x': other_x, 'z': other_z})
        return self.__class__(sel, other.cellsize, other.dz)


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
