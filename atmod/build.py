import numpy as np
import xarray as xr
from typing import Union, TypeVar

from atmod.base import Raster, VoxelModel, Mapping, AtlansParameters, AtlansStrat
from atmod.merge import combine_data_sources
from atmod.preprocessing import (
    get_numba_mapping_dicts_from,
    soilmap_to_raster,
    map_geotop_strat,
    map_nl3d_strat,
)
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


def create_atlantis_variables(voxelmodel, parameters, glg=None):
    if glg is not None:
        voxelmodel['phreatic_level'] = (glg.dims, glg.values)

    voxelmodel['rho_bulk'] = _calc_rho_bulk(voxelmodel, parameters)
    voxelmodel['zbase'] = xr.full_like(voxelmodel['surface'], parameters.modelbase)

    voxelmodel['max_oxidation_depth'] = xr.full_like(
        voxelmodel['surface'], parameters.max_oxidation_depth
    )
    voxelmodel['no_oxidation_thickness'] = xr.full_like(
        voxelmodel['surface'], parameters.no_oxidation_thickness
    )
    voxelmodel['no_shrinkage_thickness'] = xr.full_like(
        voxelmodel['surface'], parameters.no_shrinkage_thickness
    )

    bottom_holocene = voxelmodel.select_bottom(
        voxelmodel['geology'] == AtlansStrat.holocene
    )
    voxelmodel['domainbase'] = bottom_holocene.ds

    return voxelmodel


def build_atlantis_model(
    ahn: Raster,
    geotop: VoxelModel,
    nl3d: VoxelModel,
    bodemkaart: Mapping,
    parameters: AtlansParameters,
    glg: Raster = None,
):
    """
    Workflow to create a 3D subsurface model as input for Atlantis subsurface
    modelling.

    Parameters
    ----------
    ahn : Raster
        _description_
    geotop : VoxelModel
        _description_
    nl3d : VoxelModel
        _description_
    bodemkaart : Mapping
        _description_
    parameters : AtlansParameters
        _description_
    glg : Raster
        _description_

    Returns
    -------
    xr.Dataset
        _description_

    """
    soilmap_dicts = get_numba_mapping_dicts_from(bodemkaart)
    soilmap = soilmap_to_raster(bodemkaart, ahn)

    geotop = map_geotop_strat(geotop)
    nl3d = map_nl3d_strat(nl3d)

    voxelmodel = combine_data_sources(
        ahn, geotop, nl3d, soilmap, soilmap_dicts, parameters
    )
    voxelmodel = create_atlantis_variables(voxelmodel, parameters, glg)

    voxelmodel = voxelmodel.ds
    voxelmodel = voxelmodel.rename(
        {'z': 'layer'}
    )  # With renamed dimension it is no longer a vald atmod.VoxelModel  # noqa: E501
    voxelmodel['layer'] = np.arange(len(voxelmodel['layer'])) + 1

    return voxelmodel
