from enum import StrEnum
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geost.io import Geopackage


class SoilmapLayers(StrEnum):
    SOILAREA = "soilarea"
    AREAOFPEDOLOGICALINTEREST = "areaofpedologicalinterest"
    NGA_PROPERTIES = "nga_properties"
    SOILMAP = "soilmap"
    NORMALSOILPROFILES = "normalsoilprofiles"
    NORMALSOILPROFILES_LANDUSE = "normalsoilprofiles_landuse"
    SOILHORIZON = "soilhorizon"
    SOILHORIZON_FRACTIONPARTICLESIZE = "soilhorizon_fractionparticlesize"
    SOILLAYER = "soillayer"
    SOIL_UNITS = "soil_units"
    SOILCHARACTERISTICS_BOTTOMLAYER = "soilcharacteristics_bottomlayer"
    SOILCHARACTERISTICS_TOPLAYER = "soilcharacteristics_toplayer"
    SOILAREA_NORMALSOILPROFILE = "soilarea_normalsoilprofile"
    SOILAREA_SOILUNIT = "soilarea_soilunit"
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSTOPLAYER = (
        "soilarea_soilunit_soilcharacteristicstoplayer"  # noqa: E501
    )
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSBOTTOMLAYER = (
        "soilarea_soilunit_soilcharacteristicsbottomlayer"  # noqa: E501
    )


class BroSoilmap:
    """
    Class to read and query available tables in the Geopackage of the BRO Bodemkaart.

    The Bodemkaart can be downloaded from PDOK with the following url:
    https://service.pdok.nl/bzk/bro-bodemkaart/atom/downloads/BRO_DownloadBodemkaart.gpkg

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the spatial locations of the CPT data. The "find key" per
        "bro_id" to each related tables in the Geopackage as index (index name: "fid").
    db : :class:`~geost.io.Geopackage`
        Geost Geopackage instance to handle the database connections and queries.

    Usage
    -----
    Easy instance of the BroBodemKaart with a working connection by using:
    >>> soilmap = BroBodemKaart.from_geopackage(path_to_geopackage)

    Query a table in the soilmap geopackage:
    >>> with soilmap.db:
    ...     result = soilmap.db.query("SELECT * FROM soilarea")

    """

    def __init__(self, gdf: gpd.GeoDataFrame, db: Geopackage = None):
        self.gdf = gdf
        self.db = db

    @classmethod
    def from_geopackage(cls, file: str | Path, **gpd_kwargs):
        """
        Read the geopackage of the BRO Bodemkaart with a working sqlite3 connection
        to all the layers in the geopackage.

        Parameters
        ----------
        file : str | Path
            Path to the geopackage.
        **gpd_kwargs
            See geopandas.read_file documentation.

        Returns
        -------
        BroBodemKaart
            BroBodemKaart instance.
        """
        if "fid_as_index" not in gpd_kwargs:  # Needs to retain index for db selections
            gpd_kwargs["fid_as_index"] = True

        if "layer" in gpd_kwargs:
            raise ValueError("Layer cannot be passed as a Geopandas keyword argument.")

        gdf = gpd.read_file(file, layer=SoilmapLayers.SOILAREA, **gpd_kwargs)
        db = Geopackage(file)

        return cls(gdf, db)

    def get_typical_soilprofiles(self) -> pd.DataFrame:
        query = f"""
            SELECT
                left.maparea_id,
                right.normalsoilprofile_id,
                right.lowervalue,
                right.uppervalue,
                right.layernumber,
                right.faohorizonnotation,
                right.organicmattercontent,
                right.loamcontent,
                right.lutitecontent,
                right.sandmedian
            FROM {SoilmapLayers.SOILHORIZON} AS right
            JOIN {SoilmapLayers.SOILAREA_NORMALSOILPROFILE} AS left
            ON left.normalsoilprofile_id = right.normalsoilprofile_id
            """
        with self.db:
            soilprofiles = self.db.query(query)

        return soilprofiles
