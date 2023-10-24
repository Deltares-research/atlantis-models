import numba
import numpy as np
import xarray as xr
from typing import Union
from atmod.base import Raster, VoxelModel
from atmod.bro_models.geology import Lithology


def combine_data_sources(
        ahn,
        geotop,
        nl3d,
        soilmap,
        soilmap_dicts
        ):
    voxelmodel = combine_geotop_nl3d(geotop, nl3d)
    highest_voxel = np.max(voxelmodel.mask_surface_level())
    voxelmodel.ds = voxelmodel.ds.isel(z=slice(None, highest_voxel+5))

    thickness = get_full_like(voxelmodel, 0.5, np.nan)
    mass_organic = get_full_like(voxelmodel, 0.0)
    mass_organic[voxelmodel['lithok'].values==Lithology.organic] = 50.0

    return voxelmodel


def get_full_like(
        model: Union[Raster, VoxelModel],
        fill_value: float,
        invalid_value: float = None,
        dtype='float32'
        ):
    result = np.full(model.shape, fill_value, dtype=dtype)
    result[~model.isvalid] = invalid_value
    return result


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
def combine_voxels_and_soilmap(ahn, lithology, soilmap):
    ysize, xsize = ahn.shape
    invalid_ahn = np.isnan(ahn) or ahn > 95

    for i in range(ysize):
        for j in range(xsize):
            invalid_voxels = np.all(np.isnan(lithology[i, j, :]))

            if invalid_ahn or invalid_voxels:
                continue
    return
