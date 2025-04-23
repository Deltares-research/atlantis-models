import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, box

## local imports for GeoTOP examples
from atmod.bro_models import GeoTop, Lithology, StratGeoTop

xmin, xmax = 127_000, 132_000
ymin, ymax = 448_000, 453_000

bbox = (xmin, ymin, xmax, ymax)
cross_section_line = LineString([[xmin + 500, ymin + 500], [xmax - 500, ymax - 500]])

geotop = GeoTop.from_netcdf(
    r"p:\430-tgg-data\Geotop\geotop_v16\geotop_v16.nc",
    bbox=bbox,
    data_vars=["strat", "lithok"],
)

strat_select = geotop["strat"].sel(z=-0.5)
litho_select = geotop["lithok"].sel(z=-0.5)

sand = [Lithology.fine_sand, Lithology.medium_sand, Lithology.coarse_sand]

is_channel_map = strat_select.isin(StratGeoTop.channel_belts.values)
is_sand_map = litho_select.isin(sand)

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8, 4), tight_layout=True)
is_channel_map.plot(ax=ax[0])
is_sand_map.plot(ax=ax[1])
ax[0].set_title("Is channel-belt at -0.5 m NAP")
ax[1].set_title("Is sand at -0.5 m NAP")
plt.close()


section = geotop.select_with_line(cross_section_line, dist=100)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 4), tight_layout=True)
section["strat"].sel(z=slice(-20, 5)).plot(ax=ax[0])
section["lithok"].sel(z=slice(-20, 5)).isin(sand).plot(ax=ax[1])
plt.show()


channels = geotop.select_top(geotop["strat"].isin(StratGeoTop.channel_belts.values))


fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
channels.ds.plot(ax=ax)


print(2)
