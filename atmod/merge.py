import numba
import numpy as np
import xarray as xr
from typing import Union
from atmod.base import Raster, VoxelModel
from atmod.bro_models.geology import Lithology
from atmod.templates import get_full_like


def combine_data_sources(ahn, geotop, nl3d, soilmap, soilmap_dicts):
    voxelmodel = combine_geotop_nl3d(geotop, nl3d)
    # _mask_depth not as a class function because selection is too specific  # noqa: E501
    voxelmodel = _mask_depth(voxelmodel)

    thickness = get_full_like(voxelmodel, 0.5, np.nan)
    lithoclass = voxelmodel['lithok'].values
    mass_organic = get_full_like(voxelmodel, 0.0)
    mass_organic[lithoclass == Lithology.organic] = 50.0

    ahn_values = ahn.ds.values

    thickness, lithology, organic = combine_voxels_and_soilmap(
        ahn_values,
        thickness,
        lithoclass,
        mass_organic,
        soilmap.ds.values,
        soilmap_dicts.thickness,
        soilmap_dicts.lithology,
        soilmap_dicts.organic,
        voxelmodel.zmin,
    )
    voxelmodel['lithology'] = (('y', 'x', 'z'), lithology)
    voxelmodel['thickness'] = (('y', 'x', 'z'), thickness)
    voxelmodel['organic'] = (('y', 'x', 'z'), organic)
    voxelmodel['surface'] = (('y', 'x'), ahn_values)

    voxelmodel.drop_vars(['lithok'])
    return voxelmodel


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

    if np.all(geotop.isvalid_area):
        combined = geotop.ds
    else:
        combined = xr.where(geotop.isvalid_area, geotop.ds, nl3d.ds)

    return VoxelModel(combined, geotop.cellsize, geotop.dz)


@numba.njit
def combine_voxels_and_soilmap(
    ahn,
    thickness,
    lithology,
    organic,
    soilmap,
    soilmap_thickness,
    soilmap_lithology,
    soilmap_organic,
    modelbase,
):
    ysize, xsize = ahn.shape
    no_soil_map = np.int16(0)
    for i in range(ysize):
        for j in range(xsize):
            voxel_lithology = lithology[i, j, :]
            voxel_thickness = thickness[i, j, :]
            voxel_organic = organic[i, j, :]

            surface = ahn[i, j]

            invalid_surface = np.isnan(surface) or surface > 95
            invalid_voxel_column = np.all(np.isnan(voxel_lithology))

            if invalid_surface or invalid_voxel_column:
                continue

            soilnr = np.int16(soilmap[i, j])

            surface_level_voxels = modelbase + np.nansum(voxel_thickness)
            surface_difference = surface - surface_level_voxels

            if surface_difference > 2:
                vt, vl, vo = _fill_anthropogenic(
                    voxel_thickness, voxel_lithology, voxel_organic, surface_difference
                )

            elif soilnr == no_soil_map or _top_is_anthropogenic(voxel_lithology):
                if surface_level_voxels > surface:
                    vt, vl, vo = _shift_voxel_surface_down(
                        voxel_thickness,
                        voxel_lithology,
                        voxel_organic,
                        surface,
                        modelbase,
                    )
                elif surface_level_voxels < surface:
                    vt, vl, vo = _shift_voxel_surface_up(
                        voxel_thickness,
                        voxel_lithology,
                        voxel_organic,
                        surface,
                        modelbase,
                    )

            else:
                vt, vl, vo = _combine_with_soilprofile(
                    voxel_thickness,
                    voxel_lithology,
                    voxel_organic,
                    soilmap_thickness[soilnr],
                    soilmap_lithology[soilnr],
                    soilmap_organic[soilnr],
                    surface,
                    modelbase,
                )

            thickness[i, j, :] = vt
            lithology[i, j, :] = vl
            organic[i, j, :] = vo

    return thickness, lithology, organic


@numba.njit
def _fill_anthropogenic(thickness, lithology, organic, difference):
    anthropogenic = 0.0
    idx_to_fill = _get_top_voxel_idx(thickness) + 1

    thickness[idx_to_fill] = difference
    lithology[idx_to_fill] = anthropogenic
    organic[idx_to_fill] = anthropogenic
    return thickness, lithology, organic


@numba.njit
def _shift_voxel_surface_down(thickness, lithology, organic, surface, modelbase):
    depth_voxels = modelbase + np.cumsum(thickness)

    split_idx = np.argmax((depth_voxels >= surface) | np.isnan(depth_voxels))

    new_thickness_voxel = surface - depth_voxels[split_idx - 1]

    if new_thickness_voxel > 0.1:
        thickness[split_idx] = new_thickness_voxel
        thickness[split_idx + 1 :] = np.nan
        lithology[split_idx + 1 :] = np.nan
        organic[split_idx + 1 :] = np.nan
    else:
        thickness[split_idx - 1] += new_thickness_voxel
        thickness[split_idx:] = np.nan
        lithology[split_idx:] = np.nan
        organic[split_idx:] = np.nan

    return thickness, lithology, organic


@numba.njit
def _shift_voxel_surface_up(thickness, lithology, organic, surface, modelbase):
    top_idx = _get_top_voxel_idx(thickness)
    extra_thickness = surface - (modelbase + np.nansum(thickness))

    if extra_thickness > 0.1:
        thickness[top_idx + 1] = extra_thickness
        lithology[top_idx + 1] = lithology[top_idx]
        organic[top_idx + 1] = organic[top_idx]
    else:
        thickness[top_idx] += extra_thickness

    return thickness, lithology, organic


@numba.njit
def _combine_with_soilprofile(
    thickness,
    lithology,
    organic,
    soil_thickness,
    soil_lithology,
    soil_organic,
    surface,
    modelbase,
):
    split_elevation = surface - np.sum(soil_thickness)
    depth_voxels = modelbase + np.cumsum(thickness)

    split_idx = np.argmax(depth_voxels > split_elevation)

    if split_idx == 0:
        split_idx = np.argmax(depth_voxels)

        surface_voxels = np.nanmax(depth_voxels)
        if surface_voxels < split_elevation:
            lithology[split_idx] = lithology[split_idx - 1]
            organic[split_idx] = organic[split_idx - 1]

    depth_voxel_below_split = depth_voxels[split_idx - 1]
    thickness_new_voxel = split_elevation - depth_voxel_below_split

    if thickness_new_voxel < 0.01:
        soil_thickness[0] += thickness_new_voxel
        min_idx_soil = split_idx
    else:
        thickness[split_idx] = thickness_new_voxel
        min_idx_soil = split_idx + 1

    max_idx_soil = min_idx_soil + len(soil_thickness)

    thickness[min_idx_soil:max_idx_soil] = soil_thickness
    lithology[min_idx_soil:max_idx_soil] = soil_lithology
    organic[min_idx_soil:max_idx_soil] = soil_organic

    if max_idx_soil < len(thickness):
        thickness[max_idx_soil:] = np.nan
        lithology[max_idx_soil:] = np.nan
        organic[max_idx_soil:] = np.nan

    return thickness, lithology, organic


@numba.njit
def _get_top_voxel_idx(voxels):
    valid_voxels = ~np.isnan(voxels)
    if np.all(valid_voxels):
        return -1
    else:
        return np.argmax(np.nonzero(valid_voxels)[0])


@numba.njit
def _top_is_anthropogenic(lith):
    top_lith = lith[_get_top_voxel_idx(lith)]
    return top_lith == 0


def _mask_depth(voxelmodel):
    min_idx = np.where(voxelmodel['z'].values == -30)[0][0]
    highest_voxel = np.max(voxelmodel.mask_surface_level())
    max_layers_soilmap = 9

    max_idx = highest_voxel + max_layers_soilmap
    voxelmodel.ds = voxelmodel.ds.isel(z=slice(min_idx, max_idx))

    return voxelmodel
