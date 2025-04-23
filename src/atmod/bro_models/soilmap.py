import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path, WindowsPath
from typing import Union, NamedTuple
from atmod.base import Mapping
from atmod.utils import create_connection


class BodemKaartLayers(NamedTuple):
    SOILAREA = 'soilarea'
    AREAOFPEDOLOGICALINTEREST = 'areaofpedologicalinterest'
    NGA_PROPERTIES = 'nga_properties'
    SOILMAP = 'soilmap'
    NORMALSOILPROFILES = 'normalsoilprofiles'
    NORMALSOILPROFILES_LANDUSE = 'normalsoilprofiles_landuse'
    SOILHORIZON = 'soilhorizon'
    SOILHORIZON_FRACTIONPARTICLESIZE = 'soilhorizon_fractionparticlesize'
    SOILLAYER = 'soillayer'
    SOIL_UNITS = 'soil_units'
    SOILCHARACTERISTICS_BOTTOMLAYER = 'soilcharacteristics_bottomlayer'
    SOILCHARACTERISTICS_TOPLAYER = 'soilcharacteristics_toplayer'
    SOILAREA_NORMALSOILPROFILE = 'soilarea_normalsoilprofile'
    SOILAREA_SOILUNIT = 'soilarea_soilunit'
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSTOPLAYER = 'soilarea_soilunit_soilcharacteristicstoplayer'  # noqa: E501
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSBOTTOMLAYER = 'soilarea_soilunit_soilcharacteristicsbottomlayer'  # noqa: E501


class BodemKaartColumns(NamedTuple):
    AREAID = 'maparea_id'
    PROFILEID = 'normalsoilprofile_id'


class BroBodemKaart(Mapping):
    """
    Class to read and query available tables in the Geopackage of the BRO Bodemkaart.

    The Bodemkaart can be downloaded from PDOK with the following url:
    https://service.pdok.nl/bzk/bro-bodemkaart/atom/downloads/BRO_DownloadBodemkaart.gpkg

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of the Bodemkaart with the spatial extent of the soilunits.
    connection : sqlite3.Connection
        Sqlite3 Connection object to the Bodemkaart Geopackage.

    Usage
    -----
    Easy instance of the BroBodemKaart with a working connection by using:
    >>> soilmap = BroBodemKaart.from_geopackage(path_to_geopackage)

    """
    def __init__(self, gdf, connection=None):
        Mapping.__init__(self, gdf)
        self.connection = connection

        ## tables
        self.soilarea = BodemKaartLayers.SOILAREA
        self.area_pedological_interest = BodemKaartLayers.AREAOFPEDOLOGICALINTEREST
        self.nga_properties = BodemKaartLayers.NGA_PROPERTIES
        self.soilmap = BodemKaartLayers.SOILMAP
        self.normal_soilprofiles = BodemKaartLayers.NORMALSOILPROFILES
        self.normal_soilprofiles_landuse = BodemKaartLayers.NORMALSOILPROFILES_LANDUSE
        self.soilhorizon = BodemKaartLayers.SOILHORIZON
        self.soilhorizon_fraction_particlesize = BodemKaartLayers.SOILHORIZON_FRACTIONPARTICLESIZE  # noqa: E501
        self.soillayer = BodemKaartLayers.SOILLAYER
        self.soil_units = BodemKaartLayers.SOIL_UNITS
        self.soil_characteristics_bot = BodemKaartLayers.SOILCHARACTERISTICS_BOTTOMLAYER
        self.soil_characteristics_top = BodemKaartLayers.SOILCHARACTERISTICS_TOPLAYER
        self.soilarea_normal_soilprofile = BodemKaartLayers.SOILAREA_NORMALSOILPROFILE
        self.soilarea_soilunit = BodemKaartLayers.SOILAREA_SOILUNIT
        self.soilarea_soilunit_characteristics_top = BodemKaartLayers.SOILAREA_SOILUNIT_SOILCHARACTERISTICSTOPLAYER  # noqa: E501
        self.soilarea_soilunit_characteristics_bot = BodemKaartLayers.SOILAREA_SOILUNIT_SOILCHARACTERISTICSBOTTOMLAYER  # noqa: E501

    @classmethod
    def from_geopackage(cls, gpkg_path: str | WindowsPath, **gpd_kwargs):
        """
        Read the geopackage of the BRO Bodemkaart with a working sqlite3 connection
        to all the layers in the geopackage.

        Parameters
        ----------
        gpkg_path : str | WindowsPath
            Path to the geopackage.
        **gpd_kwargs
            See geopandas.read_file documentation.

        Returns
        -------
        BroBodemKaart
            BroBodemKaart instance.
        """
        gdf = gpd.read_file(gpkg_path, layer=BodemKaartLayers.SOILAREA, **gpd_kwargs)
        conn = create_connection(gpkg_path)
        return cls(gdf, conn)

    def get_cursor(self):
        return self.connection.cursor()

    def get_column_names(self, table : str) -> list:
        """
        Get the column names of a table in the BRO Bodemkaart geopackage.

        Parameters
        ----------
        table : string
            Name of the table to get the column names for.

        Returns
        -------
        columns : list
            List of the column names for the table.

        """
        cursor = self.get_cursor()
        cursor.execute(f"SELECT * FROM {table}")
        columns = [col[0] for col in cursor.description]
        return columns

    def table_head(self, table: str) -> pd.DataFrame:
        """
        Select the first five records from a table the BRO Bodemkaart geopackage.

        Parameters
        ----------
        table : string
            Name of the table to select the first records from.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the first five records.

        """
        cursor = self.get_cursor()
        cursor.execute(f'SELECT * FROM {table} LIMIT 5')
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=self.get_column_names(table))

    def query(self, query: str, outcolumns: list = None) -> pd.DataFrame:
        """
        Use a custom query on the BRO Bodemkaart geopackage to retrieve desired
        tables.

        Parameters
        ----------
        query : str
            Full string of the SQL query to retrieve the desired table with.
        outcolumns : list, optional
            Specify column names to be used for the output table. The default is
            None.

        Returns
        -------
        pd.DataFrame
            Result DataFrame of the query.

        """
        cursor = self.get_cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=outcolumns)

    def get_typical_soilprofiles(self) -> pd.DataFrame:
        left_table = self.soilarea_normal_soilprofile
        right_table = self.soilhorizon

        map_id = BodemKaartColumns.AREAID
        join_key = BodemKaartColumns.PROFILEID
        left_join_key = f'{left_table}.{join_key}'
        right_join_key = f'{right_table}.{join_key}'

        query = (
            f'SELECT {left_table}.{map_id}, {right_table}.* '
            f'FROM {left_table} '
            f'JOIN {right_table} ON {left_join_key}={right_join_key}'
        )
        cursor = self.get_cursor()
        cursor.execute(query)
        data = cursor.fetchall()

        right_columns = self.get_column_names(right_table)
        columns = [map_id] + right_columns

        return pd.DataFrame(data, columns=columns)
