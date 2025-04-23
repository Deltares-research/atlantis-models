import numpy as np
import pytest
import xarray as xr

import atmod


@pytest.fixture
def raster():
    da = xr.DataArray(
        [
            [0.61, 0.5, 0.5, 0.8, 0.52],
            [0.2, 0.36, 0.9, 0.78, 0.67],
            [0.07, 0.64, 0.01, 0.43, 0.44],
            [0.1, 0.28, 0.0, 0.44, 0.73],
            [0.93, 0.03, 0.48, 0.26, 0.63],
        ],
        dims=["y", "x"],
        coords={
            "y": np.arange(5, 0, -1) - 0.5,
            "x": np.arange(5) + 0.5,
        },
    )
    da.rio.write_crs(28992, inplace=True)
    return atmod.Raster(da)


@pytest.fixture
def raster_file(raster, tmp_path):
    outfile = tmp_path / "raster.tif"
    raster.to_raster(outfile)
    return outfile


@pytest.fixture
def xarray_dataset():
    x = np.arange(4) + 0.5
    y = x[::-1]
    z = np.arange(-2, 0, 0.5) + 0.25

    strat = [
        [[2, 2, 2, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 1, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
    ]
    lith = [
        [[2, 3, 2, 1], [2, 3, 1, 1], [2, 1, 1, 1], [3, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 3, 2, 1]],
        [[2, 3, 2, 1], [2, 1, 1, 3], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 3], [3, 2, 2, 1]],
    ]
    ds = xr.Dataset(
        data_vars=dict(strat=(["y", "x", "z"], strat), lith=(["y", "x", "z"], lith)),
        coords=dict(y=y, x=x, z=z),
    )
    return ds


@pytest.fixture
def netcdf_file(xarray_dataset, tmp_path):
    outfile = tmp_path / "test.nc"
    xarray_dataset.to_netcdf(outfile)
    return outfile


@pytest.fixture
def voxelmodel(xarray_dataset):
    return atmod.VoxelModel(xarray_dataset)


@pytest.fixture
def test_subsurface_model():
    nlayers = 4
    layers = np.arange(nlayers) + 1
    x = [150, 250, 350]
    y = [350, 250]

    nx, ny = len(x), len(y)

    data = dict(
        lithology=(["layer", "y", "x"], np.full((nlayers, ny, nx), 5)),
        thickness=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        mass_fraction_organic=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        geology=(["layer", "y", "x"], np.full((nlayers, ny, nx), 1)),
        rho_bulk=(["layer", "y", "x"], np.full((nlayers, ny, nx), 833.0)),
        mass_fraction_lutum=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        shrinkage_degree=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.7)),
        surface_level=(["y", "x"], np.full((ny, nx), 0)),
        phreatic_level=(["y", "x"], np.full((ny, nx), -0.75)),
        zbase=(["y", "x"], np.full((ny, nx), -5)),
        domainbase=(["y", "x"], np.full((ny, nx), -2)),
        no_oxidation_thickness=(["y", "x"], np.full((ny, nx), 0.0)),
        no_shrinkage_thickness=(["y", "x"], np.full((ny, nx), 0.0)),
        max_oxidation_depth=(["y", "x"], np.full((ny, nx), -1.2)),
    )

    coords = dict(
        layer=(["layer"], layers),
        y=(["y"], y),
        x=(["x"], x),
    )

    return xr.Dataset(data, coords)
