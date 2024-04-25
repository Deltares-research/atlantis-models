import pytest
import numpy as np
from numpy.testing import assert_array_equal

from atmod.build_forcings import surcharge_like


class TestSurcharge:
    @pytest.fixture
    def test_simple_surcharge_input(self):
        lithology = 5
        thickness = 0.5
        times = np.datetime64('2020-01-01').astype('datetime64[ns]')
        return lithology, thickness, times

    @pytest.fixture
    def test_surcharge_profile_input(self):
        lithology = np.array([5, 3])
        thickness = np.array([0.5, 0.2])
        times = np.arange('2020-01-01', '2024-01-01', dtype='datetime64[Y]')
        return lithology, thickness, times

    @pytest.mark.unittest
    def test_build_simple_surcharge_like(
        self, test_subsurface_model, test_simple_surcharge_input
    ):
        lithology, thickness, times = test_simple_surcharge_input

        surcharge = surcharge_like(test_subsurface_model, lithology, thickness, times)

        nl, nt = 1, 1
        nx, ny = len(test_subsurface_model['x']), len(test_subsurface_model['y'])

        assert surcharge.dims['time'] == nt
        assert surcharge.dims['layer'] == nl
        assert surcharge.dims['y'] == ny
        assert surcharge.dims['x'] == nx

        assert surcharge['lithology'].dims == ('time', 'layer', 'y', 'x')
        assert surcharge['thickness'].dims == ('time', 'layer', 'y', 'x')

    @pytest.mark.unittest
    def test_build_surcharge_profile_like(
        self, test_subsurface_model, test_surcharge_profile_input
    ):
        lithology, thickness, times = test_surcharge_profile_input

        surcharge = surcharge_like(test_subsurface_model, lithology, thickness, times)

        nl, nt = 2, 4
        nx, ny = len(test_subsurface_model['x']), len(test_subsurface_model['y'])

        assert surcharge.dims['time'] == nt
        assert surcharge.dims['layer'] == nl
        assert surcharge.dims['y'] == ny
        assert surcharge.dims['x'] == nx

        assert surcharge['lithology'].dims == ('time', 'layer', 'y', 'x')
        assert surcharge['thickness'].dims == ('time', 'layer', 'y', 'x')

        # Check if original sorting of lithology and thickness has been kept.
        assert_array_equal(surcharge['lithology'].isel(time=0, y=0, x=0), lithology)
        assert_array_equal(surcharge['thickness'].isel(time=0, y=0, x=0), thickness)
