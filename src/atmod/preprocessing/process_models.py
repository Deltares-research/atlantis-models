import numpy as np
from geost.bro import StratGeotop

from atmod.base import AtlansStrat
from atmod.bro_models import GeoTop, Nl3d, StratNl3d


def map_geotop_strat(geotop: GeoTop) -> GeoTop:
    is_holocene = geotop["strat"].isin(
        np.r_[StratGeotop.holocene, StratGeotop.channel, StratGeotop.antropogenic]
    )

    new = np.where(geotop.isvalid & is_holocene, AtlansStrat.HOLOCENE, geotop["strat"])
    new = np.where(geotop.isvalid & ~is_holocene, AtlansStrat.OLDER, new)

    geotop["strat"].values = new

    return geotop


def map_nl3d_strat(nl3d: Nl3d) -> Nl3d:
    is_holocene = nl3d["strat"].isin(StratNl3d.HOLOCEEN)

    new = np.where(nl3d.isvalid & is_holocene, AtlansStrat.HOLOCENE, nl3d["strat"])
    new = np.where(nl3d.isvalid & ~is_holocene, AtlansStrat.OLDER, new)

    nl3d["strat"].values = new

    return nl3d
