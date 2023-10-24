from dataclasses import dataclass


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
