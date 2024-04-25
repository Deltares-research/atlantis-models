import pytest
import numpy as np
import xarray as xr


@pytest.fixture
def test_subsurface_model():
    nlayers = 4
    layers = np.arange(nlayers) + 1
    x = [150, 250, 350]
    y = [350, 250]

    nx, ny = len(x), len(y)

    data = dict(
        lithology=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 5)),
        thickness=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 0.5)),
        mass_fraction_organic=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 0.5)),
        geology=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 1)),
        rho_bulk=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 833.0)),
        mass_fraction_lutum=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 0.5)),
        shrinkage_degree=(['layer', 'y', 'x'], np.full((nlayers, ny, nx), 0.7)),
        surface_level=(['y', 'x'], np.full((ny, nx), 0)),
        phreatic_level=(['y', 'x'], np.full((ny, nx), -0.75)),
        zbase=(['y', 'x'], np.full((ny, nx), -5)),
        domainbase=(['y', 'x'], np.full((ny, nx), -2)),
        no_oxidation_thickness=(['y', 'x'], np.full((ny, nx), 0.0)),
        no_shrinkage_thickness=(['y', 'x'], np.full((ny, nx), 0.0)),
        max_oxidation_depth=(['y', 'x'], np.full((ny, nx), -1.2)),
    )

    coords = dict(
        layer=(['layer'], layers),
        y=(['y'], y),
        x=(['x'], x),
    )

    return xr.Dataset(data, coords)
