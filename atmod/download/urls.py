import requests

url = (
    r'https://service.pdok.nl/rws/ahn3/wcs/v1_0?'
    r'service=wcs'
    r'&VERSION=1.0.0'
    r'&REQUEST=GetCoverage'
    r'&FORMAT=GEOTIFF_FLOAT32'
    r'&COVERAGE=ahn3_05m_dtm'
    r'&crs=EPSG:28992'
    r'&response_crs=EPSG:28992'
    r'&BBOX=88013.7238158045802265,441478.6266455391887575,90700.2187192221754231,442123.9031226849183440'
    r'&WIDTH=2685'
    r'&HEIGHT=644'
)

new_url = (
    r'https://service.pdok.nl/rws/ahn/wcs/v1_0?'
    r'service=wcs'
    r'&VERSION=1.0.0'
    r'&REQUEST=GetCoverage'
    r'&FORMAT=GEOTIFF_FLOAT32'
    r'&COVERAGE=dsm_05m'
    r'&crs=EPSG:28992'
    r'&response_crs=EPSG:28992'
    r'&BBOX=88013.7238158045802265,441478.6266455391887575,90700.2187192221754231,442123.9031226849183440'
    r'&WIDTH=2685'
    r'&HEIGHT=644'
)
