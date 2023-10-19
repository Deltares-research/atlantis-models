import numpy as np
import xarray as xr
from typing import Union

from atmod.base import Raster, VoxelModel, Mapping
from atmod.preprocessing import get_numba_mapping_dicts_from, soilmap_to_raster
from atmod.utils import build_template


def get_2d_template_like(model: Union[Raster, VoxelModel]) -> Raster:
    xmin_center = model.xmin + (0.5*model.cellsize)
    ymin_center = model.ymin + (0.5*model.cellsize)

    return build_template(
        model.ncols,
        model.nrows,
        xmin_center,
        ymin_center,
        model.cellsize
    )


def combine_geotop_nl3d(geotop: VoxelModel, nl3d: VoxelModel) -> VoxelModel:
    """
    Combine the GeoTop and NL3D voxelmodels from BRO/DINOloket. Locations where
    GeoTop is missing voxel stacks, NL3D data is filled.

    Parameters
    ----------
    geotop : VoxelModel
        atmod.bro_models.GeoTop class instance of the GeoTop voxelmodel.
    nl3d : VoxelModel
        atmod.bro_models.Nl3d class instance of the NL3D voxelmodel.

    Returns
    -------
    VoxelModel
        Combined VoxelModel instance of GeoTop and NL3D.
    """
    nl3d = nl3d.select_like(geotop)

    if np.all(geotop.isvalid):
        combined = geotop.ds
    else:
        combined = xr.where(geotop.isvalid, geotop.ds, nl3d.ds)

    return VoxelModel(combined, geotop.cellsize, geotop.dz)


def build_atlantis_model(
        ahn: Raster,
        geotop: VoxelModel,
        nl3d: VoxelModel,
        bodemkaart: Mapping,
        ):
    voxelmodel = combine_geotop_nl3d(geotop, nl3d)

    template_2d = get_2d_template_like(geotop)

    soilmap_dicts = get_numba_mapping_dicts_from(bodemkaart)
    soilmap = soilmap_to_raster(bodemkaart, template_2d)

    return soilmap_dicts, soilmap


if __name__ == "__main__":
    from atmod.bro_models import BroBodemKaart, GeoTop, Nl3d

    bbox = (200_000, 435_000, 210_000, 445_000)

    path_gpkg = r'c:\Users\knaake\OneDrive - Stichting Deltares\Documents\data\dino\bro_bodemkaart.gpkg'  # noqa: E501

    ahn = 1
    geotop = GeoTop.from_netcdf(
        r'p:\430-tgg-data\Geotop\geotop2023\geotop.nc',
        bbox=bbox,
        data_vars=['strat', 'lithok']
        )
    nl3d = Nl3d.from_netcdf(
        r'p:\430-tgg-data\NL3D\nl3d.nc',
        bbox=bbox,
        data_vars=['strat', 'lithok']
        )
    soilmap = BroBodemKaart.from_geopackage(path_gpkg, bbox=bbox)

    model = build_atlantis_model(1, geotop, nl3d, soilmap)
