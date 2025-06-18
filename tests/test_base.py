import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import atmod


class TestVoxelModel:
    @pytest.mark.unittest
    def test_voxelmodel_init(self, voxelmodel):
        assert isinstance(voxelmodel, atmod.VoxelModel)
        assert isinstance(voxelmodel.ds, xr.Dataset)
        assert voxelmodel.shape == (4, 4, 4)
        assert voxelmodel.sizes == {"y": 4, "x": 4, "z": 4}
        assert voxelmodel.bounds == (0.0, 0.0, 4.0, 4.0)
        assert voxelmodel.bounds == (0.0, 0.0, 4.0, 4.0)
        assert voxelmodel.vertical_bounds == (-2, 0)
        assert voxelmodel.xmin == 0.0
        assert voxelmodel.ymin == 0.0
        assert voxelmodel.xmax == 4.0
        assert voxelmodel.ymax == 4.0
        assert voxelmodel.zmin == -2.0
        assert voxelmodel.zmax == 0.0
        assert voxelmodel.resolution == (1.0, 1.0, 0.5)
        assert voxelmodel.z_ascending
        assert_array_equal(voxelmodel.data_vars, ["strat", "lith"])

    @pytest.mark.unittest
    def test_fail_voxelmodel_init(self, xarray_dataset):
        ds = xarray_dataset.rename_dims({"x": "lon", "y": "lat"})
        with pytest.raises(ValueError):
            atmod.VoxelModel(ds)

    @pytest.mark.unittest
    def test_from_netcdf(self, netcdf_file):
        voxelmodel = atmod.VoxelModel.from_netcdf(netcdf_file)
        assert isinstance(voxelmodel, atmod.VoxelModel)
        assert voxelmodel.shape == (4, 4, 4)
        assert voxelmodel.sizes == {"y": 4, "x": 4, "z": 4}
        assert voxelmodel.bounds == (0.0, 0.0, 4.0, 4.0)
        assert voxelmodel.vertical_bounds == (-2, 0)

        voxelmodel = atmod.VoxelModel.from_netcdf(netcdf_file, bbox=(0, 0, 2, 2))
        assert isinstance(voxelmodel, atmod.VoxelModel)
        assert voxelmodel.bounds == (0.0, 0.0, 2.0, 2.0)
        assert voxelmodel.shape == (2, 2, 4)
        assert voxelmodel.vertical_bounds == (-2, 0)
        assert voxelmodel.xmin == 0.0
        assert voxelmodel.ymin == 0.0
        assert voxelmodel.xmax == 2.0
        assert voxelmodel.ymax == 2.0
