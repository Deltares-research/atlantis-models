import numpy as np
import pytest

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
        thickness = np.repeat([0.5, np.nan], [12, 3])
        return thickness

    @pytest.fixture
    def test_voxel_lithology(self):
        lithology = np.repeat([4, 2, 1, 3, np.nan], [4, 2, 4, 2, 3])
        return lithology

    @pytest.fixture
    def test_voxel_organic(self):
        organic = np.repeat([50, np.nan], [12, 3])
        return organic

    @pytest.mark.unittest
    def test_voxel_columns_match(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        assert len(test_voxel_thickness) == 15
        assert len(test_voxel_lithology) == 15
        assert len(test_voxel_organic) == 15

    @pytest.mark.unittest
    def test_get_top_voxel_idx(self, test_voxel_thickness):
        top_idx = _get_top_voxel_idx(test_voxel_thickness)
        assert top_idx == 11

    @pytest.mark.unittest
    def test_fill_anthropogenic(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        difference = 3.1
        vt, vl, vo = _fill_anthropogenic(
            test_voxel_thickness, test_voxel_lithology, test_voxel_organic, difference
        )
        filled_index = 12
        assert vt[filled_index] == difference
        assert vl[filled_index] == 0.0
        assert vo[filled_index] == 0.0

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert np.all(remaining_nans == 2)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up_small_thickness(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 1.05

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert np.isclose(thickness_to_shift, 0.05)

        vt, _, _ = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_voxel_idx = 11
        assert np.isclose(vt[changed_voxel_idx], 0.55)

        new_surface_level = modelbase + np.nansum(vt)
        assert np.isclose(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 1.15

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert np.isclose(thickness_to_shift, 0.15)

        vt, vl, vo = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_voxel_idx = 12

        assert np.isclose(vt[changed_voxel_idx], 0.15)
        assert vl[changed_voxel_idx] == vl[changed_voxel_idx - 1]
        assert vo[changed_voxel_idx] == vo[changed_voxel_idx - 1]

        new_surface_level = modelbase + np.nansum(vt)
        assert np.isclose(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down_small_thickness(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 0.55

        vt, vl, vo = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_idx = 10
        assert np.isclose(vt[changed_idx], 0.55)
        assert np.all(np.isnan(vt[changed_idx + 1 :]))
        assert np.all(np.isnan(vl[changed_idx + 1 :]))
        assert np.all(np.isnan(vo[changed_idx + 1 :]))

        new_surface_level = modelbase + np.nansum(vt)
        assert np.isclose(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down(
        self, test_voxel_thickness, test_voxel_lithology, test_voxel_organic
    ):
        modelbase = -5.0
        surface = 0.75

        vt, _, _ = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_lithology,
            test_voxel_organic,
            surface,
            modelbase,
        )
        changed_idx = 11
        assert np.isclose(vt[changed_idx], 0.25)

        new_surface_level = modelbase + np.nansum(vt)
        assert np.isclose(new_surface_level, surface)

    @pytest.mark.unittest
    def test_combine_with_soilprofile(self):
        pass
