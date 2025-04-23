import numba
import numpy as np
import xarray as xr

from atmod.base import Raster, VoxelModel
from atmod.bro_models.geology import Lithology
from atmod.templates import get_full_like

TYPEMIN_INT64 = np.iinfo(np.int64).min


def combine_data_sources(
    ahn, geotop, parameters, nl3d=None, soilmap=None, soilmap_dicts=None
):
    if nl3d is not None:
        voxelmodel = combine_geotop_nl3d(geotop, nl3d)
    else:
        voxelmodel = geotop

    # _mask_depth not as a class function because selection is too specific  # noqa: E501
    voxelmodel = _mask_depth(voxelmodel, parameters)

    if soilmap is None:
        soilmap_values = np.full(ahn.shape, TYPEMIN_INT64)
    else:
        soilmap_values = soilmap.ds.values

    max_layers_soilmap = 9
    voxelmodel = _allocate_memory_for_soilmap(voxelmodel, max_layers_soilmap)

    thickness = get_full_like(voxelmodel, 0.5, np.nan)
    geology = voxelmodel["strat"].values
    lithoclass = voxelmodel["lithok"].values
    mass_organic = get_full_like(voxelmodel, 0.0)
    mass_organic[lithoclass == Lithology.organic] = parameters.mass_fraction_organic

    thickness, geology, lithology, organic = combine_voxels_and_soilmap(
        ahn.values,
        thickness,
        geology,
        lithoclass,
        mass_organic,
        soilmap_values,
        soilmap_dicts.thickness,
        soilmap_dicts.lithology,
        soilmap_dicts.organic,
        voxelmodel.zmin,
    )

    voxelmodel["geology"] = (("y", "x", "z"), geology)
    voxelmodel["lithology"] = (("y", "x", "z"), lithology)
    voxelmodel["thickness"] = (("y", "x", "z"), thickness)
    voxelmodel["mass_fraction_organic"] = (("y", "x", "z"), organic)
    voxelmodel["surface_level"] = (ahn.dims, ahn.values)

    voxelmodel.drop_vars(["strat", "lithok"])
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
    elif not np.any(geotop.isvalid_area):
        combined = nl3d.ds
    else:
        combined = xr.where(geotop.isvalid_area, geotop.ds, nl3d.ds)

    return VoxelModel(combined, geotop.cellsize, geotop.dz, geotop.epsg)


def _allocate_memory_for_soilmap(voxelmodel, nlayers):
    extra_zcoords = np.arange(0, voxelmodel.dz * nlayers, voxelmodel.dz) + voxelmodel.dz
    extra_zcoords = extra_zcoords + np.max(voxelmodel.zcoords)

    new_zcoords = np.append(voxelmodel.zcoords, extra_zcoords)

    voxelmodel.ds = voxelmodel.ds.reindex({"z": new_zcoords})
    return voxelmodel


@numba.njit
def combine_voxels_and_soilmap(
    ahn,
    thickness,
    geology,
    lithology,
    organic,
    soilmap,
    soilmap_thickness,
    soilmap_lithology,
    soilmap_organic,
    modelbase,
):
    ysize, xsize = ahn.shape
    no_soil_map = TYPEMIN_INT64
    for i in range(ysize):
        for j in range(xsize):
            voxel_thickness = thickness[i, j, :]
            voxel_geology = geology[i, j, :]
            voxel_lithology = lithology[i, j, :]
            voxel_organic = organic[i, j, :]

            surface = ahn[i, j]

            invalid_surface = np.isnan(surface) or surface > 95
            invalid_voxels = np.isnan(voxel_lithology)
            invalid_voxel_column = np.all(invalid_voxels)

            if invalid_surface or invalid_voxel_column:
                continue

            soilnr = np.int64(soilmap[i, j])

            # Surface level voxels will be underestimated when base of voxels is invalid
            if invalid_voxels[0]:
                first_valid = np.min(np.nonzero(~invalid_voxels)[0])

                # Add thicknesses, geology and lithologies at invalid locations.
                voxel_thickness[:first_valid] = 0.5
                voxel_geology[:first_valid] = voxel_geology[first_valid]
                voxel_lithology[:first_valid] = voxel_lithology[first_valid]

            surface_level_voxels = modelbase + np.nansum(voxel_thickness)

            surface_difference = surface - surface_level_voxels

            if surface_difference > 2:
                vt, vg, vl, vo = _fill_anthropogenic(
                    voxel_thickness,
                    voxel_geology,
                    voxel_lithology,
                    voxel_organic,
                    surface_difference,
                )

            elif soilnr == no_soil_map or _top_is_anthropogenic(voxel_lithology):
                if surface_level_voxels > surface:
                    vt, vg, vl, vo = _shift_voxel_surface_down(
                        voxel_thickness,
                        voxel_geology,
                        voxel_lithology,
                        voxel_organic,
                        surface,
                        modelbase,
                    )
                elif surface_level_voxels < surface:
                    vt, vg, vl, vo = _shift_voxel_surface_up(
                        voxel_thickness,
                        voxel_geology,
                        voxel_lithology,
                        voxel_organic,
                        surface,
                        modelbase,
                    )

            else:
                vt, vg, vl, vo = _combine_with_soilprofile(
                    voxel_thickness,
                    voxel_geology,
                    voxel_lithology,
                    voxel_organic,
                    soilmap_thickness[soilnr].copy(),
                    soilmap_lithology[soilnr].copy(),
                    soilmap_organic[soilnr].copy(),
                    surface,
                    modelbase,
                )

            thickness[i, j, :] = vt
            geology[i, j, :] = vg
            lithology[i, j, :] = vl
            organic[i, j, :] = vo

    return thickness, geology, lithology, organic


@numba.njit
def _fill_anthropogenic(thickness, geology, lithology, organic, difference):
    anthropogenic = 0.0
    idx_to_fill = _get_top_voxel_idx(thickness) + 1

    thickness[idx_to_fill] = difference
    geology[idx_to_fill] = geology[idx_to_fill - 1]
    lithology[idx_to_fill] = anthropogenic
    organic[idx_to_fill] = anthropogenic
    return thickness, geology, lithology, organic


@numba.njit
def _shift_voxel_surface_down(
    thickness, geology, lithology, organic, surface, modelbase
):
    depth_voxels = modelbase + np.cumsum(thickness)

    split_idx = np.argmax((depth_voxels >= surface) | np.isnan(depth_voxels))

    new_thickness_voxel = surface - depth_voxels[split_idx - 1]

    if new_thickness_voxel > 0.1:
        thickness[split_idx] = new_thickness_voxel
        thickness[split_idx + 1 :] = np.nan
        geology[split_idx + 1 :] = np.nan
        lithology[split_idx + 1 :] = np.nan
        organic[split_idx + 1 :] = np.nan
    else:
        thickness[split_idx - 1] += new_thickness_voxel
        thickness[split_idx:] = np.nan
        geology[split_idx:] = np.nan
        lithology[split_idx:] = np.nan
        organic[split_idx:] = np.nan

    return thickness, geology, lithology, organic


@numba.njit
def _shift_voxel_surface_up(thickness, geology, lithology, organic, surface, modelbase):
    top_idx = _get_top_voxel_idx(thickness)
    extra_thickness = surface - (modelbase + np.nansum(thickness))

    if extra_thickness > 0.1:
        thickness[top_idx + 1] = extra_thickness
        geology[top_idx + 1] = geology[top_idx]
        lithology[top_idx + 1] = lithology[top_idx]
        organic[top_idx + 1] = organic[top_idx]
    else:
        thickness[top_idx] += extra_thickness

    return thickness, geology, lithology, organic


@numba.njit
def _combine_with_soilprofile(
    thickness,
    geology,
    lithology,
    organic,
    soil_thickness,
    soil_lithology,
    soil_organic,
    surface,
    modelbase,
):
    split_elevation = surface - np.sum(soil_thickness)
    depth_voxels = modelbase + np.nancumsum(thickness)
    depth_voxels[np.isnan(thickness)] = np.nan

    split_idx = np.argmax(depth_voxels > split_elevation)

    if split_idx == 0:  # Bottom of soilprofile is above the highest voxel.
        split_idx = np.argmax(depth_voxels)

        surface_voxels = np.nanmax(depth_voxels)
        if surface_voxels < split_elevation:  # fill with voxel if it is 'empty space'
            geology[split_idx] = geology[split_idx - 1]
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

    holocene = 1.0
    older = 2.0
    top_idx_geology = _get_top_voxel_idx(geology)
    if geology[top_idx_geology] == holocene:
        geology[min_idx_soil:max_idx_soil] = holocene
    else:
        geology[min_idx_soil:max_idx_soil] = older

    thickness[min_idx_soil:max_idx_soil] = soil_thickness
    lithology[min_idx_soil:max_idx_soil] = soil_lithology
    organic[min_idx_soil:max_idx_soil] = soil_organic

    if max_idx_soil < len(thickness):
        thickness[max_idx_soil:] = np.nan
        geology[max_idx_soil:] = np.nan
        lithology[max_idx_soil:] = np.nan
        organic[max_idx_soil:] = np.nan

    return thickness, geology, lithology, organic


@numba.njit
def _get_top_voxel_idx(voxels):
    valid_voxels = ~np.isnan(voxels)
    if np.all(valid_voxels):
        return -1
    else:
        return np.max(np.nonzero(valid_voxels)[0])


@numba.njit
def _top_is_anthropogenic(lith):
    top_lith = lith[_get_top_voxel_idx(lith)]
    return top_lith == 0


def _mask_depth(voxelmodel, parameters):
    base, top = parameters.modelbase, parameters.modeltop

    if parameters.modeltop == "infer":
        min_idx, max_idx = _infer_depth_idxs(voxelmodel, base)
        voxelmodel.ds = voxelmodel.ds.isel(z=slice(min_idx, max_idx))

    else:
        if voxelmodel.z_ascending:
            voxelmodel.ds = voxelmodel.ds.sel(z=slice(base, top))
        else:
            voxelmodel.ds = voxelmodel.ds.sel(z=slice(top, base))

    return voxelmodel


def _infer_depth_idxs(voxelmodel, base):
    if voxelmodel.z_ascending:
        min_idx = np.argmax(voxelmodel["z"].values > base)
    else:
        min_idx = np.argmin(voxelmodel["z"].values > base)

    max_idx = np.max(voxelmodel.get_surface_level_mask())

    return min_idx, max_idx
