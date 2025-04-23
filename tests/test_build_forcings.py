import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from atmod.build_forcings import stage_indexation_from, surcharge_like


@pytest.fixture
def single_date():
    return np.datetime64("2020-01-01").astype("datetime64[ns]")


@pytest.fixture
def array_of_dates():
    return np.arange("2020-01-01", "2024-01-01", dtype="datetime64[Y]")


class TestSurcharge:
    @pytest.fixture
    def test_simple_surcharge_input(self, single_date):
        lithology = 5
        thickness = 0.5
        times = single_date
        return lithology, thickness, times

    @pytest.fixture
    def test_surcharge_profile_input(self, array_of_dates):
        lithology = np.array([5, 3])
        thickness = np.array([0.5, 0.2])
        times = array_of_dates
        return lithology, thickness, times

    @pytest.mark.unittest
    def test_build_simple_surcharge_like(
        self, test_subsurface_model, test_simple_surcharge_input
    ):
        lithology, thickness, times = test_simple_surcharge_input

        surcharge = surcharge_like(test_subsurface_model, lithology, thickness, times)

        nl, nt = 1, 1
        nx, ny = len(test_subsurface_model["x"]), len(test_subsurface_model["y"])

        assert surcharge.dims["time"] == nt
        assert surcharge.dims["layer"] == nl
        assert surcharge.dims["y"] == ny
        assert surcharge.dims["x"] == nx

        assert surcharge["lithology"].dims == ("time", "layer", "y", "x")
        assert surcharge["thickness"].dims == ("time", "layer", "y", "x")

    @pytest.mark.unittest
    def test_build_surcharge_profile_like(
        self, test_subsurface_model, test_surcharge_profile_input
    ):
        lithology, thickness, times = test_surcharge_profile_input

        surcharge = surcharge_like(test_subsurface_model, lithology, thickness, times)

        nl, nt = 2, 4
        nx, ny = len(test_subsurface_model["x"]), len(test_subsurface_model["y"])

        assert surcharge.dims["time"] == nt
        assert surcharge.dims["layer"] == nl
        assert surcharge.dims["y"] == ny
        assert surcharge.dims["x"] == nx

        assert surcharge["lithology"].dims == ("time", "layer", "y", "x")
        assert surcharge["thickness"].dims == ("time", "layer", "y", "x")

        # Check if original sorting of lithology and thickness has been kept.
        assert_array_equal(surcharge["lithology"].isel(time=0, y=0, x=0), lithology)
        assert_array_equal(surcharge["thickness"].isel(time=0, y=0, x=0), thickness)


class TestStageIndexation:
    @pytest.fixture
    def test_weir_areas(self):
        weir_areas = np.array([[1, 1, 2], [1, 2, 2]])
        x = [150, 250, 350]
        y = [350, 250]
        return xr.DataArray(weir_areas, coords={"y": y, "x": x}, dims=("y", "x"))

    @pytest.fixture
    def test_weir_areas_shifted_coords(self):
        weir_areas = np.array([[1, 1, 2], [1, 2, 2]])
        x = [155, 255, 355]
        y = [355, 255]
        return xr.DataArray(weir_areas, coords={"y": y, "x": x}, dims=("y", "x"))

    @pytest.fixture
    def test_spatial_factor(self):
        return np.array([[0.5, 0.5, 1.0], [0.5, 1.0, 1.0]])

    @pytest.mark.unittest
    def test_stage_indexation_from(
        self, test_subsurface_model, test_weir_areas, array_of_dates
    ):
        factor = 0.5

        si = stage_indexation_from(
            test_weir_areas, test_subsurface_model, array_of_dates, factor
        )

        nt = len(array_of_dates)
        nx, ny = len(test_subsurface_model["x"]), len(test_subsurface_model["y"])

        assert_array_equal(list(si.data_vars), ["weir_area", "factor"])

        assert len(si["time"]) == nt
        assert len(si["y"]) == ny
        assert len(si["x"]) == nx

        for date in array_of_dates:
            assert_array_equal(si["weir_area"].sel(time=date), test_weir_areas)

        assert np.all(si["factor"] == factor)

    @pytest.mark.unittest
    def test_stage_indexation_from_shifted_and_spatial_factor(
        self,
        test_subsurface_model,
        test_weir_areas_shifted_coords,
        test_spatial_factor,
        single_date,
    ):
        expected_xcoords = [150, 250, 350]
        expected_ycoords = [350, 250]

        si = stage_indexation_from(
            test_weir_areas_shifted_coords,
            test_subsurface_model,
            single_date,
            test_spatial_factor,
        )

        assert len(si["time"]) == 1
        assert_array_equal(si["x"], expected_xcoords)
        assert_array_equal(si["y"], expected_ycoords)
        assert_array_equal(si["weir_area"].isel(time=0), test_weir_areas_shifted_coords)
        assert_array_equal(si["factor"].isel(time=0), test_spatial_factor)
