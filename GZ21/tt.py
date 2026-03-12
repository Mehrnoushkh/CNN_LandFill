import xarray as xr
import numpy as np

ds = xr.open_dataset('./../updated_data/res005/data2023_01deg.nc')
T = ds['analysed_sst'].values

# Check temperature range
T_valid = T[T != -999]  # Remove land
print(f"T min: {np.min(T_valid)}")
print(f"T max: {np.max(T_valid)}")
print(f"T mean: {np.mean(T_valid)}")
