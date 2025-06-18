import dask.array as darray
import numpy as np
import xarray as xr

from atmod.base import VoxelModel


def dask_output_model_like(
    voxelmodel: VoxelModel, chunksize: int, add_phreatic=False, add_soilmap_layers=True
):
    ny, nx, nz = voxelmodel.shape

    out_zcoords = voxelmodel["z"].values

    if add_soilmap_layers:
        max_layers_soilmap = 9
        nz += max_layers_soilmap

        _, _, dz = voxelmodel.resolution
        extra_zcoords = np.arange(0, dz * max_layers_soilmap, dz)
        extra_zcoords += voxelmodel.zmax + 0.5 * dz

        out_zcoords = np.append(out_zcoords, extra_zcoords)

    empty_3d = darray.empty(
        shape=(ny, nx, nz), dtype="float64", chunks=(chunksize, chunksize, nz)
    )
    empty_2d = darray.empty(
        shape=(ny, nx), dtype="float64", chunks=(chunksize, chunksize)
    )

    data = dict(
        geology=(["y", "x", "z"], empty_3d),
        lithology=(["y", "x", "z"], empty_3d),
        thickness=(["y", "x", "z"], empty_3d),
        mass_fraction_organic=(["y", "x", "z"], empty_3d),
        surface_level=(["y", "x"], empty_2d),
        rho_bulk=(["y", "x", "z"], empty_3d),
        zbase=(["y", "x"], empty_2d),
        max_oxidation_depth=(["y", "x"], empty_2d),
        no_oxidation_thickness=(["y", "x"], empty_2d),
        no_shrinkage_thickness=(["y", "x"], empty_2d),
        domainbase=(["y", "x"], empty_2d),
    )
    if add_phreatic:
        data["phreatic_level"] = (["y", "x"], empty_2d)

    coords = dict(y=voxelmodel["y"], x=voxelmodel["x"], z=out_zcoords)

    return xr.Dataset(data, coords)


def get_full_like(
    model: VoxelModel,
    fill_value: float,
    invalid_value: float = None,
    dtype="float32",
):
    result = np.full(model.shape, fill_value, dtype=dtype)
    result[~model.isvalid] = invalid_value
    return result
