import xarray as xr


ds = xr.open_dataarray(r'p:\11200933-cas-bodemdaling\KEA_2021_bodemdalingsvoorspellingskaarten\data\02-interim\soil_map.nc')


#%% Determine the highest occurring lithology in Geotop

ds = xr.open_dataset(r'p:\430-tgg-data\Geotop\geotop2023\geotop.nc')
ds = ds.sel(x=slice(130_000, 140_000), y=slice(440_000, 450_000))

z = ds.z.values

ds.lithology.cumsum('z').argmax('z').max().values.item() + 5


#%%
import numpy as np

np.random.seed(0)

base_gtp = -5
mv = 0.705

thickness_gtp = np.full(15, 0.5)
lith_gtp = np.random.randint(1, 4, 15)
thickness_gtp[10:] = 0
lith_gtp[10:] = -1

thickness_bodem = np.array([0.2, 0.3, 0.2, 0.5])
lith_bodem = np.random.randint(1, 4, 4)



cut_level = mv - np.sum(thickness_bodem)

topdepth_gtp = base_gtp + np.cumsum(thickness_gtp)

split_idx = np.argmax(topdepth_gtp>=cut_level)

if split_idx == 0:
    split_idx = np.argmax(topdepth_gtp) + 1
    lith_gtp[split_idx] = lith_gtp[split_idx-1]


new_geo_voxel = cut_level - topdepth_gtp[split_idx-1]

print(thickness_gtp, lith_gtp)
if new_geo_voxel < 0.01:    # Als geotop voxel < 1 cm zou worden, verleng dan bodemvoxel een beetje ipv nieuwe minivoxel maken
    thickness_bodem[0] += new_geo_voxel
    min_idx_bodem = split_idx
    max_idx_bodem = min_idx_bodem + len(thickness_bodem)
    
else:
    thickness_gtp[split_idx] = new_geo_voxel
    min_idx_bodem = split_idx + 1
    max_idx_bodem = min_idx_bodem + len(thickness_bodem)
    

thickness_gtp[min_idx_bodem:max_idx_bodem] = thickness_bodem
lith_gtp[min_idx_bodem:max_idx_bodem] = lith_bodem    
thickness_gtp[max_idx_bodem:] = 0
lith_gtp[max_idx_bodem:] = -1
print(thickness_gtp, lith_gtp)
print((base_gtp + np.sum(thickness_gtp)).round(3) == mv)