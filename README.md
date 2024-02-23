# Atlantis Models (atmod)

atmod is a Python package to build subsurface models with the required input variables for land subsidence modelling with Atlantis Julia (found [here](https://gitlab.com/deltares/subsidence/atlans.jl)).

# Input for models:
Subsurface models are created from a combination BRO/DINO 2D/3D subsurface models and 2D grids for elevation (AHN) and groundwater table (Gemiddeld Laagste Grondwaterstand; GLG).

(Models)
- GeoTOP (found [here](https://dinodata.nl/opendap))
- NL3D (found [here](https://dinodata.nl/opendap))
- BRO Bodemkaart (download links found [here](https://www.pdok.nl/-/de-services-voor-de-bro-datasets-bodemkaart-en-geomorfologische-kaart-zijn-vernieuwd))

(Grids)
- AHN (download links found [here](https://www.pdok.nl/introductie/-/article/actueel-hoogtebestand-nederland-ahn))
- GLG: User input or use BRO GLG (found [here](https://basisregistratieondergrond.nl/inhoud-bro/registratieobjecten/modellen/model-grondwaterspiegeldiepte-wdm/))


# How to install
Currently, Atlantis-models is findable through the Python Package Index (PyPI). Installation can be installed directly from the repository with pip using the following command:
```
pip install git+https://gitlab.com/deltares/tgg-projects/atlantis_models.git
```
