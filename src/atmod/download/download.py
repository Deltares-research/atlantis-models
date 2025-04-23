import requests
from typing import Union
from pathlib import Path, WindowsPath


def bro_bodemkaart_geopackage(outputfile: Union[str, WindowsPath]):
    """
    Download the BRO Bodemkaart as a geopackage from PDOK.

    Parameters
    ----------
    outputfile : Union[str, WindowsPath]
        Path + filename to write the geopackage to. Ouput extension must be '.gpkg'.

    Returns
    -------
    outputfile : Union[str, WindowsPath]
        Path to the geopackage.

    """
    print("Download BRO Bodemkaart.")
    if Path(outputfile).suffix != '.gpkg':
        raise ValueError('Outputfile must be a GeoPackage (".gpkg")')

    bodemkaart_url = r'https://service.pdok.nl/bzk/bro-bodemkaart/atom/downloads/BRO_DownloadBodemkaart.gpkg'
    response = requests.get(bodemkaart_url)

    if response.status_code == 200:
        with open(outputfile, 'wb') as gpkg:
            print("Write GeoPackage.")
            gpkg.write(response.content)
    else:
        raise ValueError("Geopackage not found.")
    return outputfile
