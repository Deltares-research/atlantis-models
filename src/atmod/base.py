from abc import ABC, abstractmethod
from pathlib import WindowsPath
from typing import Optional

import numpy as np
import rasterio
import rioxarray as rio
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box

from atmod.utils import LineString, sample_along_line


class AtlansParameters:
    def __init__(
        self,
        modelbase: int | float = -30,
        modeltop: int | float | str = None,
        mass_fraction_organic: int | float = 0.5,
        mass_fraction_lutum: int | float = 0.5,
        rho_bulk: int | float = 833.0,
        shrinkage_degree: int | float = 0.7,
        max_oxidation_depth: int | float = 1.2,
        no_oxidation_thickness: int | float = 0.3,
        no_shrinkage_thickness: int | float = 0.0,
    ):
        self.modelbase = float(modelbase)

        if modeltop == "infer" or modeltop is None:
            self.modeltop = modeltop
        elif isinstance(modeltop, (int, float)):
            self.modeltop = float(modeltop)
        else:
            raise ValueError(
                'Illegal input for modeltop. Use int or float dtype or "infer".'
            )

        self.mass_fraction_organic = float(mass_fraction_organic)
        self.mass_fraction_lutum = float(mass_fraction_lutum)
        self.rho_bulk = float(rho_bulk)
        self.shrinkage_degree = float(shrinkage_degree)
        self.max_oxidation_depth = float(max_oxidation_depth)
        self.no_oxidation_thickness = float(no_oxidation_thickness)
        self.no_shrinkage_thickness = float(no_shrinkage_thickness)

    def __str__(self):
        params = [f"{k}: {v}" for k, v in self.__dict__.items()]
        instance = self.__class__.__name__
        params = "\n\t".join(params)
        return f"{instance}:\n\t{params}"

    @classmethod
    def from_inifile(cls):
        raise NotImplementedError


class AtlansStrat:
    holocene = 1
    older = 2


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
        cellsize: int | float,
        crs: str | int | CRS = None,
    ):
        self.ds = ds
        self.cellsize = cellsize

        if crs is not None:
            self.ds.rio.write_crs(crs)

    def __repr__(self):
        instance = f"atmod.{self.__class__.__name__}"
        dimensions = f"Dimensions: {dict(self.ds.sizes)}"
        resolution = f"Resolution (y, x): {self.cellsize, self.cellsize}"
        return f"{instance}\n{dimensions}\n{resolution}"

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
        return self.ds["x"].values

    @property
    def ycoords(self):
        return self.ds["y"].values

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
        return self.xcoords[-1] > self.xcoords[0]

    @property
    def y_ascending(self):
        return self.ycoords[-1] > self.ycoords[0]

    @property
    def dtype(self):
        return self.ds.dtype

    @property
    def values(self):
        return self.ds.values

    def select(self, indexers: Optional[dict] = None, **xr_kwargs):
        """
        Return a new object instance whose data is given by selecting index
        labels along the specified dimension(s).

        Parameters
        ----------
        indexers : Optional[dict], optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.

        """
        sel = self.ds.sel(indexers, **xr_kwargs)
        return self.__class__(sel, self.cellsize, self.crs)

    def select_idx(self, indexers: Optional[dict] = None, **xr_kwargs):
        sel = self.ds.isel(indexers, **xr_kwargs)
        return self.__class__(sel, self.cellsize, self.crs)

    def select_in_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        if not self.x_ascending:
            xmin, xmax = xmax, xmin
        if not self.y_ascending:
            ymin, ymax = ymax, ymin
        return self.select(y=slice(ymin, ymax), x=slice(xmin, xmax))

    def get_affine(self) -> tuple:
        """
        Get an affine matrix based on the 2D extent of the data.

        """
        x_rotation, y_rotation = 0.0, 0.0
        xsize, ysize = self.cellsize, -self.cellsize
        return xsize, x_rotation, self.xmin, y_rotation, ysize, self.ymax

    def set_cellsize(
        self,
        cellsize: int | float,
        resampling_method=Resampling.bilinear,
        inplace=False,
    ):
        upscale_factor = self.cellsize / cellsize
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

    def set_crs(self, crs: str | int | CRS):
        self.ds.rio.write_crs(crs, inplace=True)

    def to_raster(self, outputpath, **rio_kwargs):
        self.ds.rio.to_raster(outputpath, **rio_kwargs)


class VoxelModel(Raster):
    def __init__(
        self,
        ds: xr.Dataset,
        cellsize: int | float,
        dz: int | float,
        epsg: str | int | CRS = None,
    ):
        Raster.__init__(self, ds, cellsize, None)
        self.dz = dz
        self.epsg = epsg

    def __repr__(self):
        instance = f"atmod.{self.__class__.__name__}"
        layers = self.ds.data_vars
        dimensions = f"Dimensions: {dict(self.ds.sizes)}"
        resolution = f"Resolution (y, x, z): {self.cellsize, self.cellsize, self.dz}"
        return f"{instance}\n{layers}\n{dimensions}\n{resolution}"

    def __getitem__(self, item):
        return self.ds[item]

    def __setitem__(self, key, item):
        self.ds[key] = item

    @classmethod
    def from_tif(cls, *_):
        raise NotImplementedError("Cannot create VoxelModel from tif.")

    @property
    def zcoords(self):
        return self.ds["z"].values

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

    @property
    def z_ascending(self):
        return self.zcoords[-1] > self.zcoords[0]

    @staticmethod
    def coordinates_to_cellcenters(ds, cellsize, dz):
        ds["x"] = ds["x"] + (cellsize / 2)
        ds["y"] = ds["y"] + (cellsize / 2)
        ds["z"] = ds["z"] + (dz / 2)
        return ds

    def drop_vars(self, data_vars: str | list, inplace: bool = True):
        if inplace:
            self.ds = self.ds.drop_vars(data_vars)
        else:
            return self.__class__(self.ds.drop_vars(data_vars), self.cellsize, self.dz)

    def _get_raster_meta(self):
        affine = self.get_affine()
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": self.ncols,
            "height": self.nrows,
            "crs": CRS.from_epsg(self.epsg),
            "nodata": np.nan,
            "transform": affine,
            "count": 1,
        }
        return meta

    def zslice_to_tiff(self, layer: str, z: int | float, outputpath: str | WindowsPath):
        zslice = self.ds[layer].sel(z=z, method="nearest")

        if not self.x_ascending:
            zslice = zslice.isel(x=slice(None, None, -1))
        if self.y_ascending:
            zslice = zslice.isel(y=slice(None, None, -1))

        meta = self._get_raster_meta()

        with rasterio.open(outputpath, "w", **meta) as dst:
            dst.write(zslice.values, 1)

    @property
    def isvalid_area(self):
        """
        2D DataArray where GeoTop contains valid voxels at y, x locations.
        """
        if not hasattr(self, "_isvalid_area"):
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

    def get_surface_level_mask(self):
        max_idx_valid = self._get_indices_2d(self.isvalid, "max")
        return max_idx_valid

    def _get_indices_2d(self, da, which="max"):
        summed = np.cumsum(da.values, axis=2)
        summed[~self.isvalid.values] = -1

        if which == "max":
            idxs = np.argmax(summed, axis=2)
        elif which == "min":
            idxs = np.argmax(summed == 1, axis=2)
        else:
            raise ValueError('"which" can only be "min" or "max".')
        idxs[np.all(~da.values, axis=2)] = -1
        return idxs

    def select(self, indexers: Optional[dict] = None, **xr_kwargs):
        """
        Return a new object instance whose data is given by selecting index
        labels along the specified dimension(s).

        Parameters
        ----------
        indexers : Optional[dict], optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.

        """
        sel = self.ds.sel(indexers, **xr_kwargs)
        return self.__class__(sel, self.cellsize, self.dz, self.epsg)

    def select_idx(self, indexers: Optional[dict] = None, **xr_kwargs):
        sel = self.ds.isel(indexers, **xr_kwargs)
        return self.__class__(sel, self.cellsize, self.dz, self.epsg)

    def select_like(self, other):
        other_y = other.ycoords
        other_x = other.xcoords
        other_z = other.zcoords

        sel = self.ds.sel(y=other_y, x=other_x, z=other_z, method="nearest")
        sel = sel.assign_coords({"y": other_y, "x": other_x, "z": other_z})
        return self.__class__(sel, other.cellsize, other.dz, other.epsg)

    def select_top(self, cond):
        idxs = self._get_indices_2d(cond, which="max")
        top = self["z"].values[idxs] + (0.5 * self.dz)
        top[(~self.isvalid_area) | (idxs == -1)] = np.nan

        top = xr.DataArray(
            top, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )

        return Raster(top, self.cellsize, self.epsg)

    def select_bottom(self, cond):
        idxs = self._get_indices_2d(cond, which="min")
        bottom = self["z"].values[idxs] - (0.5 * self.dz)
        bottom[(~self.isvalid_area) | (idxs == -1)] = np.nan

        bottom = xr.DataArray(
            bottom, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )

        return Raster(bottom, self.cellsize, self.epsg)

    def select_with_line(
        self,
        line: LineString,
        dist: int | float = None,
        nsamples: int = None,
        cut_edges: bool = True,
    ):
        """
        Use a Shapely LineString to select data along the x and y dims of the VoxelModel
        for the creation of cross-sections. Sampling can be done using a specified dist-
        ance or a specified number of samples that need to be taken along the line.

        Parameters
        ----------
        line : LineString
            shapely.geometry.LineString object to use for sampling.
        dist : int, float, optional
            Distance between each sample along the line. Takes equally distant samples
            from the start of the line untill it reaches the end. The default is None.
        nsamples : int, optional
            Number of samples to take along the line between the beginning and the end.
            The default is None.
        cut_edges : bool, optional
            Specify whether to only sample where the line intersects with the bounding
            box of the VoxelModel. If True, parts of the line that are outside removed.

        Raises
        ------
        ValueError
            If both or none of dist and nsamples are specified.

        Returns
        -------
        ds_sel : xr.Dataset or xr.DataArray
            2D Xarray Dataset with the VoxelModel variables with new dimension 'dist'
            for distance.

        """
        if cut_edges:
            bbox = box(*self.bounds)
            line = line.intersection(bbox)

        section = sample_along_line(self.ds, line, dist, nsamples)
        return section

    def select_surface_level(self):
        surface_idx = self.get_surface_level_mask()
        surface = self.zcoords[surface_idx] + (0.5 * self.dz)
        surface = xr.DataArray(
            surface, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )
        return Raster(surface, self.cellsize, self.epsg)


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
