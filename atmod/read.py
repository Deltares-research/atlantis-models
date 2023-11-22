from atmod.base import Raster


def read_ahn(ahn_path, bbox: tuple = None):
    ahn = Raster.from_tif(ahn_path, bbox)
    # TODO: add automatic resampling if cellsize of ahn is not 100x100 m
    return ahn


def read_glg(glg_path, bbox: tuple = None):
    glg = Raster.from_tif(glg_path, bbox)
    glg.set_crs(28992)
    glg.set_cellsize(100, inplace=True)
    return glg
