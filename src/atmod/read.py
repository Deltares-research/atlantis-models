import rioxarray as rio

from atmod.utils import set_cellsize


def read_ahn(ahn_path, bbox: tuple = None):
    ahn = rio.open_rasterio(ahn_path, masked=True).squeeze(drop=True)
    if bbox is not None:
        ahn = ahn.rio.clip_box(*bbox)
    # TODO: add automatic resampling if cellsize of ahn is not 100x100 m
    return ahn


def read_glg(glg_path, bbox: tuple = None):
    glg = rio.open_rasterio(glg_path, masked=True).squeeze(drop=True)
    glg.rio.write_crs(28992, inplace=True)

    if bbox is not None:
        glg = glg.rio.clip_box(*bbox)

    glg = set_cellsize(glg, 100, 100)
    return glg
