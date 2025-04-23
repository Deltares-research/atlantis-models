from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
import rioxarray as rio
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box

from atmod.mixins import XarrayMixin
from atmod.utils import (
    LineString,
    check_dims,
    sample_along_line,
)  # TODO: remove LineString import


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


class Raster(XarrayMixin):
    def __init__(self, ds: xr.DataArray):
        self.ds = ds
        self._xdim = ds.rio._x_dim
        self._ydim = ds.rio._y_dim

    def __repr__(self):
        instance = f"atmod.{self.__class__.__name__}"
        dimensions = f"Dimensions: {dict(self.ds.sizes)}"
        resolution = f"Resolution (x, y): {self.cellsize}"
        return f"{instance}\n{dimensions}\n{resolution}"

    def __getitem__(self, item):
        return self.ds[item]

    @classmethod
    def from_tif(cls, tif_path, bbox=None):
        ds = rio.open_rasterio(tif_path).squeeze(drop=True)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

        return cls(ds)

    @property
    def shape(self) -> tuple:
        return self.ds.shape

    @property
    def x_ascending(self):
        return self[self._xdim][-1] > self[self._xdim][0]

    @property
    def y_ascending(self):
        return self[self._ydim][-1] > self[self._ydim][0]

    def select_in_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        if not self.x_ascending:
            xmin, xmax = xmax, xmin
        if not self.y_ascending:
            ymin, ymax = ymax, ymin
        return self.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))

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


class VoxelModel(XarrayMixin):
    def __init__(self, ds: xr.Dataset):
        required_dims = {"y", "x", "z"}
        if not required_dims.issubset(ds.dims):
            raise ValueError(
                f"Dataset must contain dimensions {required_dims}, "
                f"but found {set(ds.dims)}."
            )
        self.ds = ds

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
    def from_netcdf(
        cls,
        nc_path: str | Path,
        data_vars: Iterable[str] = None,
        bbox: tuple[float, float, float, float] = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read data from a NetCDF file of a voxelmodel data into a VoxelModel instance.

        This assumes the voxelmodel is according to the following conventions:
        - The coordinate dimensions of the voxelmodel are: "x", "y" (horizontal) and "z"
        (depth).
        - The coordinates in the y-dimension are in descending order.

        Parameters
        ----------
        nc_path : str | Path
            Path to the netcdf file of the voxelmodel.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            the voxelmodel. The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        VoxelModel
            VoxelModel instance of the netcdf file.

        Examples
        --------
        Read all model data from a local NetCDF file:

        >>> VoxelModel.from_netcdf("my_netcdf_file.nc")

        Read specific data variables and the data within a specific area from the NetCDF
        file:

        >>> VoxelModel.from_netcdf(
        ...     "my_netcdf_file.nc",
        ...     data_vars=["my_var"],
        ...     bbox=(1, 1, 3, 3) # (xmin, ymin, xmax, ymax)
        ... )

        Note that this method assumes the y-coordinates are in descending order. For y-
        ascending coordinates change ymin and ymax coordinates:

        >>> VoxelModel.from_netcdf(
        ...     "my_netcdf_file.nc", bbox=(1, 3, 1, 3) # (xmin, ymax, xmax, ymin)
        ... )

        """
        ds = xr.open_dataset(nc_path, **xr_kwargs)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        return cls(ds)

    def _get_internal_zbounds(self):
        self._dz = np.abs(np.diff(self["z"])[0])
        self._zmin = np.min(self["z"].values)
        self._zmax = np.max(self["z"].values)
        self._zmin -= 0.5 * self._dz
        self._zmax += 0.5 * self._dz

    @property
    def vertical_bounds(self) -> tuple[float, float]:
        if not hasattr(self, "_zmin"):
            self._get_internal_zbounds()
        return float(self._zmin), float(self._zmax)

    @property
    def zmin(self) -> float:
        return self.vertical_bounds[0]

    @property
    def zmax(self) -> float:
        return self.vertical_bounds[1]

    @property
    def resolution(self) -> tuple[float, float, float]:
        """
        Return a tuple (dy, dx, dz) of the VoxelModel resolution.
        """
        if not hasattr(self, "_dz"):
            self._get_internal_zbounds()
        dx, dy = np.abs(self.cellsize)
        return (float(dy), float(dx), float(self._dz))

    @property
    def shape(self) -> tuple:
        return tuple(self.sizes.values())

    @property
    def data_vars(self):
        return list(self.ds.data_vars.keys())

    @property
    def z_ascending(self):
        return self["z"][-1] > self["z"][0]

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

    def zslice_to_tiff(self, layer: str, z: int | float, outputpath: str | Path):
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


class Mapping:
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
