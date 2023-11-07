import numpy as np
import xarray as xr
from pathlib import WindowsPath
from typing import Union, TypeVar
from atmod.base import VoxelModel
from atmod.utils import _follow_gdal_conventions

ArrayLike = TypeVar('ArrayLike')


class GeoTop(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: Union[str, WindowsPath],
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        **xr_kwargs
        ):
        """
        Read the BRO GeoTop subsurface model from a netcdf dataset into a GeoTop
        VoxelModel instance. GeoTop can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : Union[str, WindowsPath]
            Path to the netcdf file of GeoTop.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of GeoTop.
            The default is None.

        Returns
        -------
        GeoTop
            GeoTop instance of the netcdf file.
        """
        cellsize = 100
        dz = 0.5

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = cls.coordinates_to_cellcenters(ds, cellsize)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz)

    @staticmethod
    def coordinates_to_cellcenters(ds, cellsize):
        ds['x'] = ds['x'] + (cellsize/2)
        ds['y'] = ds['y'] + (cellsize/2)
        return ds

    @classmethod
    def from_opendap(cls, url, bbox: tuple, data_vars: ArrayLike = None):
        """
        Download a small area of GeoTop from the OPeNDAP data server.

        Parameters
        ----------
        url : str
            Url to the netcdf file on the OPeNDAP server. See:
            https://www.dinoloket.nl/modelbestanden-aanvragen
        bbox : tuple
            xmin, ymin, xmax, ymax Rijksdriehoekstelsel coordinates of the area
            to download GeoTop for. (TODO: find max downloadsize for server)

        Returns
        -------
        xr.Dataset
            GeoTop subsurface model for the requested area.
        """
        cellsize = 100
        dz = 0.5

        xmin, ymin, xmax, ymax = bbox
        ds = xr.open_dataset(url, chunks={'y': 200, 'x': 200})
        ds = cls.coordinates_to_cellcenters(ds, cellsize)
        ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz)


class Nl3d(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: Union[str, WindowsPath],
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        **xr_kwargs
        ):
        """
        Read the NL3D subsurface model from a netcdf dataset into a Nl3d VoxelModel
        instance. NL3D can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : Union[str, WindowsPath]
            Path to the netcdf file of NL3D.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of NL3D.
            The default is None.

        Returns
        -------
        Nl3d
            Nl3d instance of the netcdf file.
        """
        ds = xr.open_dataset(nc_path, **xr_kwargs)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        ds = _follow_gdal_conventions(ds)
        return cls(ds, 250, 1.0)
