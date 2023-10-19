import rasterio
import requests
import geopandas as gpd
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from matplotlib import pyplot as plt


def write_rio_memfile(data, meta):
    with MemoryFile() as memfile:
        dataset = memfile.open(**meta)
        dataset.write(data)
        return dataset


def resample(ds, cellsize=100):
    meta = ds.meta
    xres, yres = ds.res

    new_height = int(ds.height * (xres/cellsize))
    new_width = int(ds.width * (yres/cellsize))

    resampled_array = ds.read(
        out_shape=(ds.count, new_height, new_width),
        resampling=Resampling.bilinear
    )
    new_transform = rasterio.transform.from_origin(
        ds.bounds.left,
        ds.bounds.top,
        new_width,
        new_height
    )
    meta['width'] = new_width
    meta['height'] = new_height
    meta['transform'] = new_transform

    resampled = write_rio_memfile(resampled_array, meta)
    return resampled


def rasterio_read(url):
    response = requests.get(url)
    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            test = resample(src)
            print(src)
    return


atom = r'https://service.pdok.nl/rws/ahn/atom/index.xml'
url = r'https://service.pdok.nl/rws/ahn/atom/downloads/dsm_05m/kaartbladindex.json'


response = requests.get(url)

gdf = gpd.GeoDataFrame.from_features(response.json())

kaartbladen = ['M_31HN1', 'M_31HN2', 'M_31HZ1', 'M_31HZ2']
test = rasterio_read(gdf['url'].iloc[0])

print(2)
