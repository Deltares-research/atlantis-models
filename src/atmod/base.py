from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import rioxarray as rio
import xarray as xr
from rasterio.crs import CRS
from shapely.geometry import box

from atmod.mixins import XarrayMixin
from atmod.utils import sample_along_line

type LineString = "LineString"


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


class AtlansStrat(IntEnum):
    holocene = 1
    older = 2


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
        resolution = f"Resolution (y, x, z): {self.resolution}"
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
            return self.__class__(self.ds.drop_vars(data_vars))

    def zslice_to_tiff(
        self, layer: str, z: int | float, outputpath: str | Path, **rio_kwargs
    ):
        zslice = self.ds[layer].sel(z=z, method="nearest")

        if zslice.rio.crs is None:
            zslice.rio.write_crs(self.crs, inplace=True)

        zslice.rio.to_raster(outputpath, **rio_kwargs)

    @cached_property
    def valid_area(self):
        """
        2D DataArray where GeoTop contains valid voxels at y, x locations.
        """
        return self.isvalid.any(dim="z")

    @cached_property
    def isvalid(self):
        return self.ds[self.data_vars[0]].notnull()

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
        other_y = other["y"]
        other_x = other["x"]
        other_z = other["z"]

        sel = self.ds.sel(y=other_y, x=other_x, z=other_z, method="nearest")
        sel = sel.assign_coords({"y": other_y, "x": other_x, "z": other_z})
        return self.__class__(sel)

    def select_top(self, cond):
        idxs = self._get_indices_2d(cond, which="max")
        top = self["z"].values[idxs] + (0.5 * self.dz)
        top[(~self.valid_area) | (idxs == -1)] = np.nan

        return xr.DataArray(
            top, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )

    def select_bottom(self, cond):
        idxs = self._get_indices_2d(cond, which="min")
        bottom = self["z"].values[idxs] - (0.5 * self.dz)
        bottom[(~self.valid_area) | (idxs == -1)] = np.nan

        return xr.DataArray(
            bottom, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )

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
        return xr.DataArray(
            surface, coords={"y": self.ycoords, "x": self.xcoords}, dims=("y", "x")
        )


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
