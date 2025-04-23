import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import atmod


class TestRaster:
    @pytest.mark.unittest
    def test_raster_init(self, raster):
        assert isinstance(raster, atmod.Raster)
        assert isinstance(raster.ds, xr.DataArray)
        assert raster._xdim == "x"
        assert raster._ydim == "y"
        assert raster.shape == (5, 5)
        assert raster.sizes == {"y": 5, "x": 5}
        assert raster.bounds == (0.0, 0.0, 5.0, 5.0)
        assert raster.xmin == 0.0
        assert raster.ymin == 0.0
        assert raster.xmax == 5.0
        assert raster.ymax == 5.0
        assert raster.cellsize == (1.0, -1.0)
        assert raster.nrows == 5
        assert raster.ncols == 5
        assert raster.dtype == "float64"
        assert raster.crs == 28992
        assert raster.x_ascending
        assert not raster.y_ascending

    @pytest.mark.unittest
    def test_from_tif(self, raster_file):
        raster = atmod.Raster.from_tif(raster_file)
        assert isinstance(raster, atmod.Raster)
        assert raster._xdim == "x"
        assert raster._ydim == "y"
        assert raster.shape == (5, 5)
        assert raster.sizes == {"y": 5, "x": 5}
        assert raster.bounds == (0.0, 0.0, 5.0, 5.0)

        raster = atmod.Raster.from_tif(raster_file, bbox=(0, 0, 2, 2))
        assert isinstance(raster, atmod.Raster)
        assert raster.bounds == (0.0, 0.0, 2.0, 2.0)
        assert raster.shape == (2, 2)


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
