class XarrayMixin:
    @property
    def coords(self) -> dict:
        return self.ds.coords

    @property
    def dims(self) -> tuple:
        return self.ds.dims

    @property
    def sizes(self) -> dict:
        return self.ds.sizes

    @property
    def bounds(self) -> tuple:
        return self.ds.rio.bounds()

    @property
    def xmin(self) -> float:
        return self.bounds[0]

    @property
    def ymin(self) -> float:
        return self.bounds[1]

    @property
    def xmax(self) -> float:
        return self.bounds[2]

    @property
    def ymax(self) -> float:
        return self.bounds[3]

    @property
    def cellsize(self) -> tuple:
        return self.ds.rio.resolution()

    @property
    def crs(self):
        return self.ds.rio.crs

    @property
    def nrows(self) -> int:
        return self.ds.rio.height

    @property
    def ncols(self) -> int:
        return self.ds.rio.width

    @property
    def dtype(self):
        return self.ds.dtype

    def get_affine(self):
        """
        Get the affine transformation of the dataset.

        Returns
        -------
        Affine
            Affine transformation of the dataset.

        """
        return self.ds.rio.transform()

    def sel(self, **xr_kwargs):
        """
        Use Xarray selection functionality to select indices along specified dimensions.
        This uses the ".sel" method of an Xarray Dataset.

        Parameters
        ----------
        **xr_kwargs
            xr.Dataset.sel keyword arguments. See relevant Xarray documentation.

        Examples
        --------
        Select a specified coordinates or slice coordinates from the VoxelModel instance:

        >>> selected = voxelmodel.sel(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.sel({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.sel(x=slice(1, 4))  # Using a slice

        Using additional options as well. For instance, when the desired coordinates do
        not exactly match the VoxelModel coordinates, select the nearest:

        >>> selected = voxelmodel.sel(x=[1.1, 2.5, 3.3], method="nearest")

        """
        selected = self.ds.sel(**xr_kwargs)
        return self.__class__(selected)

    def isel(self, **xr_kwargs):
        """
        Use Xarray selection functionality to select indices along specified dimensions.
        This uses the ".isel" method of an Xarray Dataset.

        Parameters
        ----------
        **xr_kwargs
            xr.Dataset.isel keyword arguments. See relevant Xarray documentation.

        Examples
        --------
        Select a specified coordinates or slice coordinates from the VoxelModel instance:

        >>> selected = voxelmodel.isel(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.isel({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.isel(x=slice(1, 4))  # Using a slice

        """
        selected = self.ds.isel(**xr_kwargs)
        return self.__class__(selected)
