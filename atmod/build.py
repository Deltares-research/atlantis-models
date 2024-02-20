import numpy as np
import xarray as xr
from typing import Union, TypeVar

from atmod.base import Raster, VoxelModel, Mapping, AtlansParameters
from atmod.merge import combine_data_sources
from atmod.preprocessing import get_numba_mapping_dicts_from, soilmap_to_raster
from atmod.read import read_ahn, read_glg
from atmod.templates import build_template
from atmod.warnings import suppress_warnings


BodemKaartDicts = TypeVar('BodemKaartDicts')


def get_2d_template_like(model: Union[Raster, VoxelModel]) -> Raster:
    xmin_center = model.xmin + (0.5 * model.cellsize)
    ymin_center = model.ymin + (0.5 * model.cellsize)

    return build_template(
        model.ncols, model.nrows, xmin_center, ymin_center, model.cellsize
    )


@suppress_warnings(RuntimeWarning)
def _calc_rho_bulk(voxelmodel, parameters):
    organic = voxelmodel['mass_fraction_organic']
    rho_bulk = (100 / organic) * (1 - np.exp(-organic / 0.12))
    rho_bulk = rho_bulk.fillna(parameters.rho_bulk).where(voxelmodel.isvalid)
    return rho_bulk


def create_atlantis_variables(voxelmodel, glg, parameters):
    voxelmodel['phreatic_level'] = (glg.dims, glg.values)
    voxelmodel['rho_bulk'] = _calc_rho_bulk(voxelmodel, parameters)
    voxelmodel['zbase'] = xr.full_like(glg.ds, voxelmodel.zmin)

    voxelmodel['max_oxidation_depth'] = xr.full_like(
        glg.ds, parameters.max_oxidation_depth
    )
    voxelmodel['no_oxidation_thickness'] = xr.full_like(
        glg.ds, parameters.no_oxidation_thickness
    )
    voxelmodel['no_shrinkage_thickness'] = xr.full_like(
        glg.ds, parameters.no_shrinkage_thickness
    )

    return voxelmodel


def build_atlantis_model(
    ahn: Raster,
    glg: Raster,
    geotop: VoxelModel,
    nl3d: VoxelModel,
    bodemkaart: Mapping,
    parameters: AtlansParameters,
):
    """
    Workflow to create a 3D subsurface model as input for Atlantis subsurface
    modelling.

    Parameters
    ----------
    ahn : Raster
        _description_
    glg : Raster
        _description_
    geotop : VoxelModel
        _description_
    nl3d : VoxelModel
        _description_
    bodemkaart : Mapping
        _description_
    parameters : AtlansParameters
        _description_

    Returns
    -------
    _type_
        _description_
    """
    soilmap_dicts = get_numba_mapping_dicts_from(bodemkaart)
    soilmap = soilmap_to_raster(bodemkaart, ahn)

    voxelmodel = combine_data_sources(
        ahn, geotop, nl3d, soilmap, soilmap_dicts, parameters
    )
    voxelmodel = create_atlantis_variables(voxelmodel, glg, parameters)

    return voxelmodel


if __name__ == "__main__":
    from atmod.bro_models import BroBodemKaart, GeoTop, Nl3d

    bbox = (200_000, 435_000, 210_000, 445_000)
    # bbox = (200_000, 435_000, 201_000, 436_000)
    path_gpkg = r'c:\Users\knaake\OneDrive - Stichting Deltares\Documents\data\dino\bro_bodemkaart.gpkg'  # noqa: E501
    path_glg = r'n:\Projects\11209000\11209259\B. Measurements and calculations\009 effectmodule bodemdaling\data\1-external\deltascenarios\S2050BP18\Modflow\GLG_19120101000000.asc'  # noqa: E501

    ahn = read_ahn(r'p:\430-tgg-data\ahn\dtm_100m.tif', bbox=bbox)
    glg = read_glg(path_glg, bbox=bbox)

    geotop = GeoTop.from_netcdf(
        r'p:\430-tgg-data\Geotop\geotop2023\geotop.nc',
        bbox=bbox,
        data_vars=['strat', 'lithok'],
        lazy=False
    )
    nl3d = Nl3d.from_netcdf(
        r'p:\430-tgg-data\NL3D\nl3d.nc',
        bbox=bbox,
        data_vars=['strat', 'lithok'],
        lazy=False
    )
    soilmap = BroBodemKaart.from_geopackage(path_gpkg, bbox=bbox)

    parameters = AtlansParameters()

    model = build_atlantis_model(ahn, glg, geotop, nl3d, soilmap, parameters)
    print(2)
