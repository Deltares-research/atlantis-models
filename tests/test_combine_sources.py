import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal

from atmod.merge import (
    _fill_anthropogenic,
    _shift_voxel_surface_down,
    _shift_voxel_surface_up,
    _combine_with_soilprofile,
    _get_top_voxel_idx,
)


class TestCombineColumns:
    @pytest.fixture
    def test_voxel_thickness(self):
        thickness = np.repeat([0.5, np.nan], [10, 5])
        return thickness

    @pytest.fixture
    def test_voxel_lithology(self):
        lithology = np.repeat([4, 2, 1, 3, np.nan], [3, 2, 3, 2, 5])
        return lithology

    @pytest.fixture
    def test_voxel_organic(self):
        organic = np.repeat([50, np.nan], [10, 5])
        return organic

    @pytest.fixture
    def test_soil_thickness(self):
        return np.array([0.2, 0.3, 0.2, 0.5])

    @pytest.fixture
    def test_soil_lithology(self):
        return np.array([1, 3, 3, 2])

    @pytest.fixture
    def test_soil_organic(self):
        return np.array([0.2, 0.3, 0.3, 0.1])

    @pytest.mark.unittest
    def test_voxel_columns_match(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        assert len(test_voxel_thickness) == 15
        assert len(test_voxel_lithology) == 15
        assert len(test_voxel_organic) == 15

    @pytest.mark.unittest
    def test_sum_soil_thickness(self, test_soil_thickness):
        assert np.sum(test_soil_thickness) == 1.2

    @pytest.mark.unittest
    def test_get_top_voxel_idx(self, test_voxel_thickness):
        top_idx = _get_top_voxel_idx(test_voxel_thickness)
        assert top_idx == 9

    @pytest.mark.unittest
    def test_fill_anthropogenic(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        difference = 3.1
        vt, vl, vo = _fill_anthropogenic(
            test_voxel_thickness, test_voxel_lithology, test_voxel_organic, difference
        )
        filled_index = 10
        assert vt[filled_index] == difference
        assert vl[filled_index] == 0.0
        assert vo[filled_index] == 0.0

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 4)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up_small_thickness(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 0.05

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert_almost_equal(thickness_to_shift, 0.05)

        vt, _, _ = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_voxel_idx = 9
        assert_almost_equal(vt[changed_voxel_idx], 0.55)

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 0.15

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert_almost_equal(thickness_to_shift, 0.15)

        vt, vl, vo = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_voxel_idx = 10

        assert_almost_equal(vt[changed_voxel_idx], 0.15)
        assert vl[changed_voxel_idx] == vl[changed_voxel_idx - 1]
        assert vo[changed_voxel_idx] == vo[changed_voxel_idx - 1]

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down_small_thickness(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = -0.45

        vt, vl, vo = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_idx = 8
        assert_almost_equal(vt[changed_idx], 0.55)
        assert np.all(np.isnan(vt[changed_idx + 1 :]))
        assert np.all(np.isnan(vl[changed_idx + 1 :]))
        assert np.all(np.isnan(vo[changed_idx + 1 :]))

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = -0.25

        vt, _, _ = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_idx = 9
        assert_almost_equal(vt[changed_idx], 0.25)

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_combine_fitting_soilprofile(
        self,
        test_voxel_thickness,
        test_voxel_lithology,
        test_voxel_organic,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.2

        vt, vl, vo = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )
        min_idx_soil = 10
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 1)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_overlapping_soilprofile(
        self,
        test_voxel_thickness,
        test_voxel_lithology,
        test_voxel_organic,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 0.45

        vt, vl, vo = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 9
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        assert_equal(vt[min_idx_soil - 1], 0.25)
        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 2)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_above(
        self,
        test_voxel_thickness,
        test_voxel_lithology,
        test_voxel_organic,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.3

        vt, vl, vo = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 11
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        assert_almost_equal(vt[min_idx_soil - 1], 0.1)
        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)

        no_remaining_nans = ~np.any(np.isnan(np.array([vt, vl, vo])))
        assert no_remaining_nans

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_above_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_lithology,
        test_voxel_organic,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.204

        vt, vl, vo = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 10
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        assert_almost_equal(vt[min_idx_soil], 0.204)
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 1)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_below_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_lithology,
        test_voxel_organic,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 0.703

        vt, vl, vo = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 9
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        assert_almost_equal(vt[min_idx_soil], 0.203)
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 2)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)
