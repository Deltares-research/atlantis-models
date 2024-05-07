import numpy as np
import xarray as xr
from pathlib import WindowsPath
from typing import TypeVar, Optional
from atmod.base import VoxelModel
from atmod.utils import _follow_gdal_conventions, get_crs_object

ArrayLike = TypeVar('ArrayLike')


class GeoTop(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | WindowsPath,
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read the BRO GeoTop subsurface model from a netcdf dataset into a GeoTop
        VoxelModel instance. GeoTop can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | WindowsPath
            Path to the netcdf file of GeoTop.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of GeoTop.
            The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.

        Returns
        -------
        GeoTop
            GeoTop instance of the netcdf file.

        """
        cellsize = 100
        dz = 0.5
        crs = 28992

        if lazy and 'chunks' not in xr_kwargs:
            xr_kwargs['chunks'] = 'auto'

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print('Load data')
            ds = ds.load()

        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz, crs)

    @classmethod
    def from_opendap(
        cls,
        url: str = r'https://dinodata.nl/opendap/GeoTOP/geotop.nc',
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Download an area of GeoTop directly from the OPeNDAP data server into a GeoTop
        VoxelModel instance.

        Parameters
        ----------
        url : str
            Url to the netcdf file on the OPeNDAP server. See:
            https://www.dinoloket.nl/modelbestanden-aanvragen
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of GeoTop.
            The default is None but for practical reasons, specifying a bounding box is
            advised (TODO: find max downloadsize for server).
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.

        Returns
        -------
        GeoTop
            GeoTop instance for the selected area.

        """
        return cls.from_netcdf(url, data_vars, bbox, lazy, **xr_kwargs)


class Nl3d(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | WindowsPath,
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read the NL3D subsurface model from a netcdf dataset into a Nl3d VoxelModel
        instance. NL3D can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | WindowsPath
            Path to the netcdf file of NL3D.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of NL3D.
            The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.

        Returns
        -------
        Nl3d
            Nl3d instance of the netcdf file.

        """
        cellsize = 250
        dz = 1.0
        crs = 28992

        if lazy and 'chunks' not in xr_kwargs:
            xr_kwargs['chunks'] = 'auto'

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print('Load data')
            ds = ds.load()

        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz, crs)

    @classmethod
    def from_opendap(
        cls,
        strat_url=r'https://dinodata.nl/opendap/NL3D/nl3d_lithostrat.nc',
        lithok_url=r'https://dinodata.nl/opendap/NL3D/nl3d_lithoklasse.nc',
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        cellsize = 250
        dz = 1.0
        crs = 28992

        if lazy and 'chunks' not in xr_kwargs:
            xr_kwargs['chunks'] = 'auto'

        ds = xr.Dataset()

        if 'lithostrat' in data_vars or data_vars is None:
            strat = xr.open_dataset(
                strat_url, drop_variables=['crs', 'lat', 'lon'], **xr_kwargs
            )

            for var in strat.data_vars:
                ds[var] = strat[var]

        if 'lithoklasse' in data_vars or data_vars is None:
            lithok = xr.open_dataset(
                lithok_url, drop_variables=['crs', 'lat', 'lon'], **xr_kwargs
            )

            for var in lithok.data_vars:
                ds[var] = lithok[var]

        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print('Load data')
            ds = ds.load()

        return cls(ds, cellsize, dz, crs)
