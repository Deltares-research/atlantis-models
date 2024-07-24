import numpy as np
from atmod.base import AtlansStrat
from atmod.bro_models import GeoTop, Nl3d, StratGeoTop, StratNl3d


def map_geotop_strat(geotop: GeoTop) -> GeoTop:
    holocene_units = np.r_[
        StratGeoTop.holocene.values,
        StratGeoTop.channel_belts.values,
        StratGeoTop.anthropogenic.values,
    ]

    is_holocene = geotop['strat'].isin(holocene_units)

    new = np.where(geotop.isvalid & is_holocene, AtlansStrat.holocene, geotop['strat'])
    new = np.where(geotop.isvalid & ~is_holocene, AtlansStrat.older, new)

    geotop['strat'].values = new

    return geotop


def map_nl3d_strat(nl3d: Nl3d) -> Nl3d:
    holocene = StratNl3d.units.select_units('Holocene eenheden').values

    is_holocene = nl3d['strat'].isin(holocene)

    new = np.where(nl3d.isvalid & is_holocene, AtlansStrat.holocene, nl3d['strat'])
    new = np.where(nl3d.isvalid & ~is_holocene, AtlansStrat.older, new)

    nl3d['strat'].values = new

    return nl3d
