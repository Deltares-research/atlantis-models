from pathlib import Path
from owslib.wcs import WebCoverageService


def main(cell_size, bbox_size):
    x = 90000.0
    y = 350000.3 # coordinaten in RD/EPSG:28992 van een locatie in NL
    origin = [10000,618750] # origin van ahn3 coverage in RD/EPSG:28992
    cell_size = cell_size # cell size van ahn3 coverage
    bbox_size = bbox_size # in meters - dit is formaat van coverage dat opgehaald wordt
    bbox_size_pixels = bbox_size / cell_size
    wcs_url = "https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&request=GetCapabilities"

    x_lower_bound = origin[0] + (((x - origin[0]) // cell_size) * cell_size) # n.b. // -> floor operator  https://www.askpython.com/python/python-floor-division-double-slash-operator
    x_upper_bound = x_lower_bound + (bbox_size_pixels * cell_size)
    y_lower_bound = origin[1] + (((y - origin[1]) // cell_size) * cell_size)
    y_upper_bound = y_lower_bound + (bbox_size_pixels * cell_size)

    wcs = WebCoverageService(wcs_url, version='2.0.1')
    output = wcs.getCoverage(
        identifier=['dsm_05m'],
        format='image/tiff',
        crs='EPSG:28992',
        subsets = [('X', x_lower_bound, x_upper_bound), ('Y', y_lower_bound, y_upper_bound)],
        width=bbox_size_pixels,
        height=bbox_size_pixels,
        scalefactor=10.0
    )
    return output


if __name__ == "__main__":
    cellsize = 0.5
    bbox_size = 60

    output = main(cellsize, bbox_size)

    workdir = Path(r'c:\Users\knaake\OneDrive - Stichting Deltares\Documents\data\ahn_test')
    with open(workdir/rf'test_{cellsize}_m_{bbox_size}_with_bbox.tif', 'wb') as f:
        f.write(output.read())



