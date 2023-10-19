import numba
import numpy as np
import xarray as xr
from atmod.base import VoxelModel


def combine_data_sources(
        ahn,
        geotop,
        nl3d,
        soilmap,
        ):
    voxelmodel = combine_geotop_nl3d(geotop, nl3d)

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

    if np.all(geotop.isvalid):
        combined = geotop.ds
    else:
        combined = xr.where(geotop.isvalid, geotop.ds, nl3d.ds)

    return VoxelModel(combined, geotop.cellsize, geotop.dz)


def combine_voxels_and_soilmap(ahn, lithology, soilmap):
    ysize, xsize = ahn.shape
    invalid_ahn = np.isnan(ahn) or ahn > 95

    for i in range(ysize):
        for j in range(xsize):
            invalid_voxels = np.all(np.isnan(lithology[i, j, :]))

            if invalid_ahn or invalid_voxels:
                continue
    return