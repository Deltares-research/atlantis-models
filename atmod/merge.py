import numba
import numpy as np


@numba.njit
def combine_data_sources(
        ahn,
        thickness,
        lithology,
        organic,
        soilmap,
        ):
    ysize, xsize = ahn.shape
    invalid_ahn = np.isnan(ahn) or ahn > 95

    for i in range(ysize):
        for j in range(xsize):
            invalid_voxels = np.all(np.isnan(lithology[i, j, :]))

            if invalid_ahn or invalid_voxels:
                continue

    return
