import rasterio
import requests
import numpy as np
import geopandas as gpd
from rasterio.merge import merge
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
    new_transform = ds.transform * ds.transform.scale(
        (ds.width / resampled_array.shape[-1]),
        (ds.height / resampled_array.shape[-2])
    )

    meta['width'] = new_width
    meta['height'] = new_height
    meta['transform'] = new_transform

    resampled = write_rio_memfile(resampled_array, meta)
    return resampled


def rasterio_read(url, cellsize=None):
    response = requests.get(url)
    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            resample(src, cellsize=cellsize)
            print(src)
    return


def is_sum_of_100(value):
    return value % 100 == 0 and value != 0


atom = r'https://service.pdok.nl/rws/ahn/atom/index.xml'
kaartblad_url = r'https://service.pdok.nl/rws/ahn/atom/downloads/dtm_05m/kaartbladindex.json'


response = requests.get(kaartblad_url)
kaartbladen = gpd.GeoDataFrame.from_features(response.json())
kaartbladen[['xmin', 'ymin', 'xmax', 'ymax']] = kaartbladen.bounds
kaartbladen.sort_values(by=['ymax', 'xmin'], inplace=True)

outtifs = []
for ii, url in enumerate(kaartbladen['url'].iloc[401:], start=401):
    print(ii, url)
    tiff_response = requests.get(url)
    with MemoryFile(tiff_response.content) as memfile:
        with memfile.open() as src:
            outtifs.append(resample(src, cellsize=5))

    if is_sum_of_100(ii) or ii == len(kaartbladen) - 1:
        print("write to tif")
        mosaic, trans = merge(outtifs)
        mosaic[mosaic==mosaic.max()] = np.nan
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': mosaic.shape[2],
            'height': mosaic.shape[1],
            'transform': trans,
            'count': 1,
            'crs': 'EPSG:28992'
        }

        outfile = rf'p:\430-tgg-data\ahn\dtm\ahn_dtm_{ii-100}_{ii}.tif'
        with rasterio.open(outfile, 'w', **meta, compress='lzw') as dst:
            dst.write(mosaic)

        outtifs = []
        print(outtifs)

print(2)
