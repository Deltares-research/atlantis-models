import numpy as np
from dataclasses import dataclass

from atmod.bro_models.nl3d_strat import NL3D_STRAT
from atmod.bro_models.gtp_strat import (
    ANTHROPOGENIC,
    HOLOCENE,
    OLDER,
    CHANNELBELTS
)


class Strat:
    def __init__(self, dict_):
        self.dict = dict_

    def __repr__(self):
        return str(self.dict)

    def __getitem__(self, item):
        return self.dict[item]

    def __getattr__(self, attr):
        return self.dict[attr]

    @property
    def names(self):
        return np.array(list(self.dict.keys()))

    @property
    def values(self):
        return np.array(list(self.dict.values()))

    def get_names(self, units):
        return self.names[np.isin(self.values, units)]

    def select_units(self, keys):
        if isinstance(keys, str):
            new_dict = {keys: self.dict[keys]}
        else:
            new_dict = {k: self.dict[k] for k in keys}
        return self.__class__(new_dict)


@dataclass
class Lithology:
    anthropogenic = 0
    organic = 1
    clay = 2
    loam = 3
    fine_sand = 5
    medium_sand = 6
    coarse_sand = 7
    gravel = 8
    shells = 9


class StratGeoTop:
    anthropogenic = Strat(ANTHROPOGENIC)
    holocene = Strat(HOLOCENE)
    older = Strat(OLDER)
    channel_belts = Strat(CHANNELBELTS)
    NN = 0


class StratNl3d:
    units = Strat(NL3D_STRAT)
