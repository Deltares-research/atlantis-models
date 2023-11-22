dimport xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import numba
from numba import types
from numba.typed import Dict
from dask import delayed
import dask.array as darray
import imod
import rasterio
from pathlib import Path

def preprocess_ondergrondmodel(ondergrondmodel_in):
    """
    Hier alleen belangrijk: ondergrondmodel thickness en mass_frac_org aangemaakt.

    Eventueel: selectie zrange nog belangrijk
    """
    z = ondergrondmodel_in.z.values
    try:
        zmax = z[ondergrondmodel_in.lithology.cumsum('z').argmax('z').max().values.item() + 5]
    except:
        zmax = z.max()   
        
    zmin = ondergrondmodel_in.z.min()
    ondergrondmodel = ondergrondmodel_in.sel(z=slice(zmin,zmax))
   
    valid = ondergrondmodel.lithology != -1
    veen = ondergrondmodel.lithology == 1
    ondergrondmodel['geology'] = xr.full_like(ondergrondmodel.lithology, 1).where(valid,-1)
    ondergrondmodel['thickness'] = xr.full_like(ondergrondmodel.lithology.astype(float),0.5).where(valid,0)
    ondergrondmodel['mass_fraction_organic'] = xr.full_like(ondergrondmodel.lithology.astype(float),0.0).where(~veen, 50)

    return(ondergrondmodel)

def create_numba_ordered_dicts(bodemdata):
    """
    Create numba ordered dictionaries to be used as lookup tables. Each soil
    code is coupled to arrays of thickness, lithology and organic mass pct
    
    """
    
    bodemdata['bodem_nr'] = bodemdata['bodem_nr'].astype(np.int16)
    
    bodemdict_orgf = Dict.empty(
        key_type=types.int16,
        value_type=types.float32[:],
    )

    bodemdict_thickness = Dict.empty(
        key_type=types.int16,
        value_type=types.float32[:],
    )    
    
    bodemdict_lith = Dict.empty(
        key_type=types.int16,
        value_type=types.int8[:],
    )
    
    bodemdict_gebr = Dict.empty(
        key_type=types.int16,
        value_type=types.int8[:],
    )
    
    grp = bodemdata.groupby('bodem_nr')
    
    for code, gr in grp:
            bodemdict_orgf[code] = np.float32(gr['org_masspct'])
            bodemdict_thickness[code] = np.float32(gr['thickness'])
            bodemdict_lith[code] = np.int8(gr['new_lith'])
            bodemdict_gebr[code] = np.int8(gr['gebruik'])
    return(bodemdict_orgf, bodemdict_thickness, bodemdict_lith, bodemdict_gebr)


@numba.njit
def combine_geotop_and_soilprofile(thickness_geo, thickness_bodem, lith_geo, lith_bodem, orgf_geo, orgf_bodem, mv):
    # Allocate nieuwe arrays
    thickness_new = np.copy(thickness_geo)
    lith_new = np.copy(lith_geo)
    orgf_new = np.copy(orgf_geo)
    
    # Flip soil_map arrays zodat ze in zelfde volgorde staan als geotop.
    thickness_bodem = np.copy(thickness_bodem[::-1])                        # NIET GEBRUIKEN VAN COPY HIER HEB IK HOOFD OVER GEBROKEN. THICKNESS BODEM WERD TELKENS BEETEJ AANGEPAST GVD
    lith_bodem = np.copy(lith_bodem[::-1])
    orgf_bodem = np.copy(orgf_bodem[::-1])
    
    # Bepalen van grensvlak tussen soil_map en geotop + afgeknipte voxeldikte
    cut_level = mv - np.sum(thickness_bodem)            # NAP hoogte van grens tussen soil_map en geotop
    geo_csum = -30+np.cumsum(thickness_geo)    
    split_idx = (geo_csum>=cut_level).argmax()          # Index langs de thicknesskolom waar geotop opgesplitst moet worden    
    if split_idx == 0:
        split_idx = geo_csum.argmax()+1
        lith_new[split_idx] = lith_new[split_idx-1]    # plaats een extra geotop voxel op een een nieuwe index
        
    new_geo_voxel = cut_level - geo_csum[split_idx-1] # dikte van geotop voxel na afsnijden
    
    # Indelen van de nieuwe waarsy
    if new_geo_voxel < 0.01:    # Als geotop voxel < 1 cm zou worden, verleng dan bodemvoxel een beetje ipv nieuwe minivoxel maken
        thickness_new[split_idx-1] = thickness_new[split_idx-1]
        thickness_bodem[0] = thickness_bodem[0]+new_geo_voxel
        thickness_new[split_idx:split_idx+len(thickness_bodem)] = thickness_bodem    
        lith_new[split_idx:split_idx+len(lith_bodem)] = lith_bodem
        orgf_new[split_idx:split_idx+len(orgf_bodem)] = orgf_bodem
        
        thickness_new[split_idx+len(thickness_bodem):] = 0.
        lith_new[split_idx+len(lith_bodem):] = -1.
        orgf_new[split_idx+len(orgf_bodem):] = -1.  
        
    else:   
        thickness_new[split_idx] = new_geo_voxel   
        thickness_new[split_idx+1:split_idx+1+len(thickness_bodem)] = thickness_bodem    
        lith_new[split_idx+1:split_idx+1+len(lith_bodem)] = lith_bodem
        orgf_new[split_idx+1:split_idx+1+len(orgf_bodem)] = orgf_bodem     
        
        thickness_new[split_idx+len(thickness_bodem)+1:] = 0.
        lith_new[split_idx+len(lith_bodem)+1:] = -1.
        orgf_new[split_idx+len(orgf_bodem)+1:] = -1.       
    return(thickness_new, lith_new, orgf_new)
    

@numba.njit
def iterate_over_coordinates(bodemdict_orgf, 
                             bodemdict_thickness, 
                             bodemdict_lith,
                             bodemdict_gebr,
                             array_thickness,
                             array_lith,
                             array_orgf,
                             soil_map,
                             landuse_map,
                             ahn3,
                             xsize, 
                             ysize):
    
    thickness_new = np.zeros_like(array_thickness).astype(np.float32)
    lith_new = np.zeros_like(array_thickness).astype(np.int8)
    orgf_new = np.zeros_like(array_thickness).astype(np.float32)    
    
    invalid_ahn = np.isnan(m) or mv >= 95 # 95 betekent: je wil niks berekenen. Over waarde kan je debaten
    invalid_geotop = np.all(array_lith==-1, axis=2)

    for i in range(ysize):
        for j in range(xsize):
            
            # print(i,j)
            # i = 95
            # j = 1
            bodem_nr = np.int16(soil_map[i,j])
            mv = ahn3[i,j]
       
            
            thickness_geo = array_thickness[i,j,:]
            lith_geo = array_lith[i,j,:]
            orgf_geo = array_orgf[i,j,:]
            
            """
            -30 meter is ondergrens waarop gtp wordt afgesneden
            """
            if np.isnan(mv) or array_lith[i,j,0]==-1 or mv >= 95:      # If there is no sifficient data at the location to complete the model  
                continue
            elif (mv - (-30 + np.sum(thickness_geo))) > 2: # als verschil ahn mv - geotop mv meer dan 2 m is
                thickness_new[i,j,0:] = thickness_geo
                lith_new[i,j,0:] = lith_geo
                orgf_new[i,j,0:] = orgf_geo

                top_idx = np.max(thickness_geo.nonzero()[0])+1                     
                thickness_new[i,j,top_idx] = (mv - (-30 + np.sum(thickness_geo))) # op top_index wordt de dikte het maaiveldverschil
                lith_new[i,j,top_idx] = 0. # dit wordt antropoceen
                orgf_new[i,j,top_idx] = 0.  
                continue                
                       
            elif bodem_nr==np.int32(0) or geotop_top_voxel == 0: # 0 is geen bodemcode gevonden of nan in bodemkaart
                thickness_new[i,j,0:] = thickness_geo
                lith_new[i,j,0:] = lith_geo
                orgf_new[i,j,0:] = orgf_geo
                if mv < (-30 + np.sum(thickness_new[i,j])):    # Als AHN3 mv onder GT mv ligt, knip bovenste Geotop voxel(s) af
                    geo_csum = -30 + np.cumsum(thickness_geo)    
                    split_idx = (geo_csum >= mv).argmax()   
                    new_geo_voxel = mv - geo_csum[split_idx-1]
                    
                    if new_geo_voxel > 0.1: # dit impliceert een minimale laagdikte van 10cm
                        thickness_new[i,j,split_idx] = new_geo_voxel
                        thickness_new[i,j,split_idx+1:] = 0.
                        lith_new[i,j,split_idx+1:] = -1.
                        orgf_new[i,j,split_idx+1:] = -1.
                    else:
                        thickness_new[i,j,split_idx-1] += new_geo_voxel
                        thickness_new[i,j,split_idx:] = 0.
                        lith_new[i,j,split_idx:] = -1.
                        orgf_new[i,j,split_idx:] = -1.
                elif mv > (-30 + np.sum(thickness_new[i,j])):  # Als AHN3 erboven ligt, plaats een extra voxel boven op geoTOP met laatst bekende Gt eenheid
                    top_idx = np.max(thickness_geo.nonzero()[0])+1
                    extra_thickness = mv - (-30 + np.sum(thickness_geo))                  
                    if extra_thickness > 0.1:
                        thickness_new[i,j,top_idx] = extra_thickness
                        lith_new[i,j,top_idx] = lith_new[i,j,top_idx-1]
                        orgf_new[i,j,top_idx] = orgf_new[i,j,top_idx-1]
                    else:
                        thickness_new[i,j,top_idx-1] += extra_thickness
                else:
                    pass                   
                continue
            
            # Use bodemgebruik to check what part of the dicts to extract
            # or to revert to a default bodem
            bodemgebruik = bodemdict_gebr[bodem_nr]
            bodems = np.unique(bodemgebruik)      
            if np.min(np.abs(bodems-gebr))==0:               # Situation: there is a match between a soil profile and the land use map. This is ideally what we want. 
                idxs = np.where(bodemdict_gebr[bodem_nr]==gebr)                
                thickness_bodem = bodemdict_thickness[bodem_nr][idxs]
                lith_bodem = bodemdict_lith[bodem_nr][idxs]
                orgf_bodem = bodemdict_orgf[bodem_nr][idxs]               
            else:
                if len(bodems) > 1:                 # Situation: there is no soil profile avail for the land use unit, but there is more than one soil profile defined. Default to the first one (with the lowest land use number) so it will revert to either 1 or 2.
                    idxs = np.where(bodemdict_gebr[bodem_nr]==np.min(bodems))                   
                    thickness_bodem = bodemdict_thickness[bodem_nr][idxs]
                    lith_bodem = bodemdict_lith[bodem_nr][idxs]
                    orgf_bodem = bodemdict_orgf[bodem_nr][idxs]
                else:                               # Situation: there is no soil profile avail for the land use unit and there is only one soil profile so use that one
                    thickness_bodem = bodemdict_thickness[bodem_nr]
                    lith_bodem = bodemdict_lith[bodem_nr]
                    orgf_bodem = bodemdict_orgf[bodem_nr]
            # print(thickness_bodem)         
            thickness_new[i,j,0:], lith_new[i,j,0:], orgf_new[i,j,0:] = create_final_column(thickness_geo, 
                                                                                        thickness_bodem, 
                                                                                        lith_geo, 
                                                                                        lith_bodem, 
                                                                                        orgf_geo, 
                                                                                        orgf_bodem, 
                                                                                        mv) 
    return(thickness_new, lith_new, orgf_new)

def _MAIN(path_ondergrondmodel_in,
          path_ahn,
          path_soil_map,
          path_landuse_map,
          path_soil_data,
          path_holocene_base,
          path_glg,
           path_control_unit,
          y_slice):
    
    # Load GeoTop
    ondergrondmodel_in = xr.open_zarr(path_ondergrondmodel_in).sel(y=y_slice)
    ahn3 = xr.load_dataarray(path_ahn).sel(y=y_slice)
    
    holocene_base = xr.load_dataarray(path_holocene_base).sel(y=y_slice) # Vervangen voor lithostrat mapping
    glg = xr.open_dataarray(path_glg).sel(y=y_slice)

    
    soil_map = xr.open_dataarray(path_soil_map).sel(y=y_slice) # = BroBodemKaart.to_mapping_table()
    soil_data = pd.read_csv(path_soil_data, sep='\t') # = BroBodemKaart.to_mapping_table()
    bodemdict_orgf, bodemdict_thickness, bodemdict_lith, bodemdict_gebr = create_numba_ordered_dicts(soil_data)  
    ondergrondmodel = preprocess_ondergrondmodel(ondergrondmodel_in)   
    
    xsize = np.int16(ondergrondmodel.dims['x'])
    ysize = np.int16(ondergrondmodel.dims['y'])
    
    array_lith = ondergrondmodel['lithology'].values
    array_thickness = ondergrondmodel['thickness'].values
    array_orgf = ondergrondmodel['mass_fraction_organic'].values
       
    thickness_new, lith_new, orgf_new = iterate_over_coordinates(bodemdict_orgf, 
                                                                bodemdict_thickness, 
                                                                bodemdict_lith,
                                                                
                                                                array_thickness, # komt uit preprocess
                                                                array_lith, # komt uit preprocess
                                                                array_orgf, # komt uit preprocess
                                                                soil_map.values,
                                                          
                                                                ahn3.values, 
                                                                xsize, 
                                                                ysize)

    model = xr.Dataset(data_vars = {'thickness': (('y', 'x', 'z'), thickness_new), 
                                              'lithology': (('y', 'x', 'z'), lith_new), 
                                              'mass_fraction_organic': (('y', 'x', 'z'), orgf_new)},
                                  coords = {'y': ondergrondmodel.y, 'x': ondergrondmodel.x, 'z': ondergrondmodel.z})
    
    valid_yxz = model.thickness.notnull()
    model["lithology"] = model["lithology"].where(valid_yxz, other=-1) 
    model['mass_fraction_organic'] = model['mass_fraction_organic'].where(model['mass_fraction_organic'] >= 0) /100
    model['mass_fraction_organic'].where(valid_yxz)
    model["geology"] = xr.full_like(model['lithology'], 1)
    model["geology"] = model["geology"].where((model["z"] > holocene_base), other = 2).where(valid_yxz, other=-1) 
    rho_bulk = (100.0 / model["mass_fraction_organic"]) * (
        1.0 - np.exp(-model["mass_fraction_organic"] / 0.12)
    )
    keep = np.isfinite(rho_bulk)
    model["rho_bulk"] = rho_bulk.where(keep).fillna(833.0).where(valid_yxz)
    model["surface_level"] = ahn3.copy()
    model["phreatic_level"] = glg.copy()
    model["aquifer_head"] = glg.copy().astype(np.float64)
    model["zbase"] = xr.full_like(model.surface_level,(ondergrondmodel.z.min()-0.25))
    model["maximum_oxidation_depth"] = xr.full_like(model.surface_level, 1.2)
    model["capillary_rise"] = xr.full_like(model.surface_level, 0.3)
    model["control_unit"] = control_unit
    model["aquifer_top"] = holocene_base
    model["minimal_mass_fraction_organic"] = xr.full_like(model.surface_level, 0.05)

    model["geology"] = model["geology"].where(model["z"] > model["aquifer_top"], other = 2) 
    model = model.rename({"z": "layer"})     
    model = model.transpose("y","x",'layer')
    model['layer'] = np.arange(1,len(ondergrondmodel.z)+1,1)

    return(model)

path_ondergrondmodel_in = snakemake.input.path_ondergrondmodel_in
path_ahn = snakemake.input.path_ahn
path_soil_map = snakemake.input.path_soil_map
path_yslices = snakemake.input.path_yslices
path_landuse_map = snakemake.input.path_landuse_map
path_soil_data = snakemake.input.path_soil_data
path_glg = snakemake.input.path_glg
path_control_unit = snakemake.input.path_control_unit
path_holocene_base = snakemake.input.path_holocene_base
paths_output = snakemake.output.path_output

if __name__ == '__main__':
    yslices = pd.read_csv(path_yslices)
    slices = []
    for index, row in yslices.iterrows():
        slices.append(slice(row.ymax,row.ymin))
       
    for i, y_slice in enumerate(slices):
        print(y_slice)
        ondergrondmodel_out =_MAIN(path_ondergrondmodel_in,
                                path_ahn,
                                path_soil_map,
                                path_landuse_map,
                                path_soil_data,
                                path_holocene_base,
                                path_glg,
                                path_control_unit[i],
                                y_slice)

        ondergrondmodel_out.to_netcdf(paths_output[i])
        del ondergrondmodel_out
                

    
   

