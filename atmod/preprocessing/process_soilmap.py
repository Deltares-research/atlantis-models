import numba
import numpy as np
import pandas as pd
from dataclasses import dataclass
from atmod.bro_models import BroBodemKaart, Lithology
from atmod.warnings import suppress_warnings


EMPTYDICT = numba.typed.Dict.empty(
    key_type=numba.types.int64,
    value_type=numba.types.float64[:],
)


@dataclass(repr=False)
class NumbaDicts:
    thickness: numba.typed.typeddict.Dict
    lithology: numba.typed.typeddict.Dict
    organic: numba.typed.typeddict.Dict
    lutum: numba.typed.typeddict.Dict

    def __repr__(self):
        keys = list(self.__dict__.keys())
        return f'NumbaDicts instance of typical soilprofiles:\n\tAttributes: {keys}'

    @classmethod
    @suppress_warnings(numba.NumbaTypeSafetyWarning)
    def from_soilmap(cls, soilmap, ascending_depth=True):
        """
        Generate Numba compatible dictionaries from the BRO Bodemkaart of layer, thickness,
        lithology and organic matter fraction for typical soilprofiles. The dictionaries
        map soilprofile id's to arrays of the soilprofile characteristics. The dictionaries
        are used to combine the Bodemkaart with the GeoTOP subsurface model of DINO-TNO.

        Parameters
        ----------
        soilmap : atmod.bro_models.BroBodemKaart
            Instance of the BRO bodemkaart from: https://www.broloket.nl/ondergrondmodellen.
        ascending_depth : bool, optional
            If True, the arrays of the soilprofiles in the dictionary will be ordered
            from top to bottom (i.e. depth ascending), the default is True.

        Returns
        -------
        NumbaDicts
            Dataclass containing numba dictionaries of thickness, lithology and organic
            matter.

        """  # noqa: E501
        lithology = EMPTYDICT
        thickness = EMPTYDICT
        organic = EMPTYDICT
        lutum = EMPTYDICT

        mapping_table = get_bodemkaart_mapping_table(soilmap)
        mapping_table['nr'] = (
            mapping_table['nr'].str.split('.', expand=True)[2].astype(int)
        )

        to_fraction = 100
        for nr, df in mapping_table.groupby('nr'):
            lith = cls._get_values(df['lithology'], ascending_depth)
            thick = cls._get_values(df['thickness'], ascending_depth)
            org = cls._get_values(df['orgmatter'] / to_fraction, ascending_depth)
            lut = cls._get_values(df['lutum'] / to_fraction, ascending_depth)

            nr = np.int64(nr)
            lithology[nr] = lith.astype('float64')
            thickness[nr] = thick.astype('float64')
            organic[nr] = org.astype('float64')
            lutum[nr] = lut.astype('float64')

        return cls(thickness, lithology, organic, lutum)

    @classmethod
    def empty(cls):
        """
        Return an instance with all empty dictionaries.

        """
        return cls(EMPTYDICT, EMPTYDICT, EMPTYDICT, EMPTYDICT)

    @staticmethod
    def _get_values(series, ascending_depth=True):
        """
        Helper function for get_numba_mapping_dicts_from.

        """
        if ascending_depth:
            values = series.values[::-1]
        else:
            values = series.values
        return values


def determine_geotop_lithology_from(soiltable: pd.DataFrame) -> np.array:
    """
    Rowwise determination of the lithology according to the GeoTop lithology classes
    based on the table of typical soilprofiles from the BRO Bodemkaart.

    Parameters
    ----------
    soiltable : pd.DataFrame
        Pandas DataFrame of the typical soilprofiles in the BRO Bodemkaart.

    Returns
    -------
    lithology : np.array
        Numpy array of the lithology classes.

    """
    soiltable['sand'] = 100 - soiltable['loamcontent']

    ## get indices of lithologies
    organic = _is_organic(soiltable)
    fine_sand, medium_sand, coarse_sand = _is_sand(soiltable)
    clay = _is_clay(soiltable)

    lithology = np.full(len(soiltable), Lithology.loam)
    lithology[organic] = Lithology.organic
    lithology[fine_sand] = Lithology.fine_sand
    lithology[medium_sand] = Lithology.medium_sand
    lithology[coarse_sand] = Lithology.coarse_sand
    lithology[clay & ~organic] = Lithology.clay
    return lithology


def _is_organic(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_geotop_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to organic

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_geotop_lithology_from'.

    """
    return (soiltable['organicmattercontent'] > 25) & (soiltable['sand'] < 65)


def _is_sand(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_geotop_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to sand.

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_geotop_lithology_from'.

    """
    is_sand = (soiltable['sand'] >= 65) & (soiltable['lutitecontent'] < 35)
    fine_sand = soiltable['sandmedian'] <= 210
    medium_sand = (soiltable['sandmedian'] > 210) & (soiltable['sandmedian'] <= 420)
    coarse_sand = soiltable['sandmedian'] > 420

    fine_sand = fine_sand & is_sand
    medium_sand = medium_sand & is_sand
    coarse_sand = coarse_sand & is_sand

    return fine_sand, medium_sand, coarse_sand


def _is_clay(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_geotop_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to clay.

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_geotop_lithology_from'.

    """
    return soiltable['lutitecontent'] > 50


def get_bodemkaart_mapping_table(soilmap) -> pd.DataFrame:
    """
    Create a Pandas DataFrame from the BRO Bodemkaart containing the location id's
    and typical soilprofile characteristics.

    Parameters
    ----------
    soilmap : atmod.bro_models.BroBodemKaart
        Instance of the BRO bodemkaart from: https://www.broloket.nl/ondergrondmodellen.

    Returns
    -------
    pd.DataFrame
        DataFrame of the typical soilprofiles.

    """
    typical_soilprofiles = soilmap.get_typical_soilprofiles()

    lithology = determine_geotop_lithology_from(typical_soilprofiles)
    thickness = typical_soilprofiles['uppervalue'] - typical_soilprofiles['lowervalue']

    mapping_table = pd.DataFrame(
        dict(
            nr=typical_soilprofiles['maparea_id'],
            soilid=typical_soilprofiles['normalsoilprofile_id'],
            soilunit=typical_soilprofiles['faohorizonnotation'],
            layer=typical_soilprofiles['layernumber'],
            thickness=thickness,
            lithology=lithology,
            orgmatter=typical_soilprofiles['organicmattercontent'],
            lutum=typical_soilprofiles['lutitecontent'],
        )
    )

    return mapping_table
