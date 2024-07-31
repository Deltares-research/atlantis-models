import numpy as np
import xarray as xr
from pathlib import WindowsPath
from typing import TypeVar

from atmod.base import Raster, VoxelModel, Mapping, AtlansParameters, AtlansStrat
from atmod.bro_models import BroBodemKaart
from atmod.merge import combine_data_sources
from atmod.preprocessing import (
    NumbaDicts,
    soilmap_to_raster,
    map_geotop_strat,
    map_nl3d_strat,
)
from atmod.templates import build_template, dask_output_model_like
from atmod.utils import find_overlapping_areas, COMPRESSION
from atmod.warnings import suppress_warnings


BodemKaartDicts = TypeVar('BodemKaartDicts')


def get_2d_template_like(model: Raster | VoxelModel) -> Raster:
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


def calculate_domainbase(voxelmodel, parameters):
    thickness_holocene = np.nansum(
        voxelmodel['thickness'].where(voxelmodel['geology'] == AtlansStrat.holocene),
        axis=2,
    )
    surface_level_voxels = parameters.modelbase + np.nansum(
        voxelmodel['thickness'], axis=2
    )
    domainbase = surface_level_voxels - thickness_holocene
    domainbase[thickness_holocene == 0] = np.nan
    return domainbase


def create_atlantis_variables(voxelmodel, parameters, glg=None):
    if glg is not None:
        voxelmodel['phreatic_level'] = (glg.dims, glg.values)

    voxelmodel['rho_bulk'] = _calc_rho_bulk(voxelmodel, parameters)
    voxelmodel['zbase'] = xr.full_like(
        voxelmodel['surface_level'], parameters.modelbase
    )

    voxelmodel['max_oxidation_depth'] = xr.full_like(
        voxelmodel['surface_level'], parameters.max_oxidation_depth
    )
    voxelmodel['no_oxidation_thickness'] = xr.full_like(
        voxelmodel['surface_level'], parameters.no_oxidation_thickness
    )
    voxelmodel['no_shrinkage_thickness'] = xr.full_like(
        voxelmodel['surface_level'], parameters.no_shrinkage_thickness
    )
    domainbase = calculate_domainbase(voxelmodel, parameters)
    voxelmodel['domainbase'] = (('y', 'x'), domainbase)

    return voxelmodel


def build_atlantis_model(
    ahn: Raster,
    geotop: VoxelModel,  # TODO: Also make input for geotop optional
    nl3d: VoxelModel = None,
    bodemkaart: Mapping = None,
    glg: Raster = None,
    parameters: AtlansParameters = None,
):
    """
    Create a 3D subsurface model that can be used as input for subsidence modelling in
    Atlantis Julia. A minimal subsurface model combines AHN (Algemeen Hoogtebestand
    Nederland) with the GeoTOP 3D voxelmodel. For a more detailed model or for areas
    where GeoTOP is unavailable, the BRO Bodemkaart and the NL3D 3D voxelmodel can be
    included. When the BRO Bodemkaart is included, the top 1.2 m below the surface level
    is replaced by the buildup as specified in the Bodemkaart.

    Parameters
    ----------
    ahn : Raster
        Raster instance with the AHN data to use for the surface level for the area
        where the model is created for.
    geotop : VoxelModel
        VoxelModel instance with the GeoTOP 3D voxelmodel data. See atmod.bro_models.GeoTop.
    bodemkaart : Mapping, optional
        Mapping instance containing the BRO Bodemkaart data. See atmod.bro_models.BroBodemKaart.
        The default is None.
    glg : Raster, optional
        Optional Raster instance with the GLG data to use for the phreatic level for the area
        where the model is created for. The default is None, then the GLG needs to be added
        manually because a phreatic level is mandatory input for an Atlantis subsurface model.
    parameters : AtlansParameters, optional
        Optional parameter class to specify default parameters for variables to use in the
        model. The default is None, then all Atlantis defaults will be used for each parameter.

    Returns
    -------
    xr.Dataset
        Xarray Dataset with the required input variables for an Atlantis subsurface model
        (N.B. mind the optional GLG input) that can be stored as a Netcdf.

    """  # noqa: E501
    if bodemkaart is not None:
        soilmap_dicts = NumbaDicts.from_soilmap(bodemkaart)
        soilmap = soilmap_to_raster(bodemkaart, ahn)
    else:
        soilmap_dicts = NumbaDicts.empty()
        soilmap = None

    geotop = map_geotop_strat(geotop)

    if nl3d is not None:
        nl3d = map_nl3d_strat(nl3d)

    if parameters is None:
        parameters = AtlansParameters()  # use all defaults

    voxelmodel = combine_data_sources(
        ahn, geotop, parameters, nl3d, soilmap, soilmap_dicts
    )
    voxelmodel = create_atlantis_variables(voxelmodel, parameters, glg)

    voxelmodel = voxelmodel.ds
    voxelmodel = voxelmodel.rename({'z': 'layer'})  # No longer a vald atmod.VoxelModel
    voxelmodel['layer'] = np.arange(len(voxelmodel['layer'])) + 1

    return voxelmodel.astype('float64')


def build_model_in_chunks(
    ahn: Raster,
    geotop: VoxelModel,
    nl3d: VoxelModel,
    bodemkaart: Mapping,
    glg: Raster,
    parameters: AtlansParameters = None,
    chunksize: int = 250,
):
    if parameters is None:
        parameters = AtlansParameters()  # use all defaults

    geotop = geotop.select(z=slice(parameters.modelbase, parameters.modeltop))

    overlapping_area = find_overlapping_areas(ahn, geotop, nl3d, glg)
    geotop = geotop.select_in_bbox(overlapping_area)

    soilmap_dicts = NumbaDicts.from_soilmap(bodemkaart)
    soilmap = soilmap_to_raster(bodemkaart, ahn)

    model = dask_output_model_like(geotop, chunksize, True, True)

    model = model.map_blocks(
        _write_model_chunk,
        kwargs={
            'ahn': ahn,
            'geotop': geotop,
            'nl3d': nl3d,
            'soilmap': soilmap,
            'soilmap_dicts': soilmap_dicts,
            'glg': glg,
            'parameters': parameters,
        },
        template=model,
    )
    model = model.rename({'z': 'layer'})
    model['layer'] = np.arange(len(model['layer'])) + 1

    return model


def _write_model_chunk(chunk, **kwargs):

    ahn = kwargs['ahn'].select(x=chunk['x'], y=chunk['y'])
    geotop = kwargs['geotop'].select(x=chunk['x'], y=chunk['y'])
    nl3d = kwargs['nl3d'].select(
        x=slice(chunk['x'].min(), chunk['x'].max()),
        y=slice(chunk['y'].max(), chunk['y'].min()),
    )
    soilmap = kwargs['soilmap'].select(x=chunk['x'], y=chunk['y'])
    soilmap_dicts = kwargs['soilmap_dicts']
    glg = kwargs['glg'].select(x=chunk['x'], y=chunk['y'])
    params = kwargs['parameters']

    geotop = map_geotop_strat(geotop)
    nl3d = map_nl3d_strat(nl3d)

    model = combine_data_sources(ahn, geotop, params, nl3d, soilmap, soilmap_dicts)
    model = create_atlantis_variables(model, params, glg)

    for var in model.data_vars:
        chunk[var].data = model[var]

    return chunk
