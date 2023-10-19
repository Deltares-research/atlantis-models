import numpy as np
from owslib.wcs import WebCoverageService


class AhnWcs:
    """
    Attempt to download ahn from the WCS server. So far does not work.

    """
    def __init__(self, url=r'https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&request=GetCapabilities'):
        self.wcs = WebCoverageService(url)
        print(2)

    @property
    def layers(self):
        return list(self.wcs.contents.keys())

    @property
    def available_formats(self):
        return self.wcs.getOperationByName('GetMap').formatOptions

    def available_crs(self, layer: str):
        return self.wcs[layer].crsOptions()

    def download_dtm(self, bbox):
        # raise NotImplementedError("No access to WCS service yet.")

        xmin, ymin, xmax, ymax = bbox
        width_pixels = (xmax - xmin) // 0.5
        height_pixels = (ymax - ymin) // 0.5
        subsets = [('X', xmin, xmax), ('Y', ymin, ymax)]

        response = self.wcs.getCoverage(
            identifier=['dtm_05m'],
            format='image/tiff',
            crs='EPSG:28992',
            subsets=subsets,
            width=width_pixels,
            height=height_pixels
        )
        # array = np.frombuffer(response.read(), np.float32)
        return response


if __name__ == "__main__":
    from atmod.build import build_template_model

    xmin, xmax = 120_000, 121_000
    ymin, ymax = 440_000, 441_000
    bbox = (xmin, ymin, xmax, ymax)

    ahn = AhnWcs()
    ahn.download_dtm(bbox)

    template = build_template_model(100, 100, 126_550, 450_050, 100)

    print(2)
