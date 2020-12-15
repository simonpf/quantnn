from datetime import datetime


from satpy import Scene
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import xarray

from pansat.products.reanalysis.era5 import ERA5Product

data_path = "/home/simon/src/pansat/notebooks/products/"
modis_files = [data_path + "MODIS/MYD021KM.A2016286.1750.061.2018062214718.hdf",
               data_path + "MODIS/MYD03.A2016286.1750.061.2018062032022.hdf"]


def prepare_input_data(modis_files,
                       pad_along_orbit=300,
                       pad_across_orbit=200):
    """
    Prepares input data for CTP retrieval from MODIS input files.

    Args:
        modis_file: List of filename containing the MODIS input data to use
            as input.
        pad_along_orbit: How many pixels to discard at the border along
            the swath.
        pad_across_orbit: How many pixels to discard at the border across
            the swath.

    Returns:
        Tuple ``(lats, lons, x)`` where ``lats`` and ``lons`` contain the
        coordinates of the input data and ``x`` the input data with the
        input features along the third dimension.
    """
    scene = Scene(filenames = modis_files, reader = 'modis_l1b')
    scene.load(["31", "32", "latitude", "longitude"], resolution=1000)

    pad_along_orbit = 300
    pad_across_orbit = 300
    di = pad_along_orbit
    dj = pad_across_orbit
    lats_r = scene["latitude"][di:-di, dj:-dj].compute()
    lons_r = scene["longitude"][di:-di, dj:-dj].compute()

    bt_11 = scene["31"][di:-di, dj:-dj].compute()
    bt_12 = scene["32"][di:-di, dj:-dj].compute()

    reduced_shape = np.array(bt_11.shape) - np.array([2, 2])
    bt_11_w = np.zeros(reduced_shape)
    bt_11_c = np.zeros(reduced_shape)
    bt_12_w = np.zeros(reduced_shape)
    bt_12_c = np.zeros(reduced_shape)
    bt_11_s = np.zeros(reduced_shape)
    bt_1112_s = np.zeros(reduced_shape)
    for i in range(1, bt_11.shape[0] - 2):
        print(i)
        for j in range(1, bt_11.shape[1] - 2):
            i_start = i - 1
            i_end = i + 1
            j_start = j - 1
            j_end = j + 1
            bt_11_n = bt_11[i_start:i_end, j_start:j_end]
            bt_12_n = bt_12[i_start:i_end, j_start:j_end]

            bt_11_w[i, j] = np.max(bt_11_n)
            bt_11_c[i, j] = np.min(bt_11_n)
            bt_11_s[i, j] = np.std(bt_11_n)

            bt_12_w[i, j] = np.max(bt_12_n)
            bt_12_c[i, j] = np.min(bt_12_n)

            bt_1112_s[i, j] = np.std(bt_11_n - bt_12_n)

    #
    # Get ERA5 data.
    #

    t_0 = datetime(2016, 10, 12, 17, 45)
    t_1 = datetime(2016, 10, 12, 17, 50)

    lat_min = scene["latitude"].data.min().compute()
    lat_max = scene["latitude"].data.max().compute()
    lon_min = scene["longitude"].data.min().compute()
    lon_max = scene["longitude"].data.max().compute()


    surface_variables = ["surface_pressure", "2m_temperature", "tcwv"]
    domain = [lat_min - 2, lat_max + 2, lon_min - 2, lon_max + 2]
    surface_product = ERA5Product('monthly',
                                'surface',
                                surface_variables,
                                domain)
    era_surface_files = surface_product.download(t_0, t_1)

    pressure_variables = ["temperature"]
    pressure_product = ERA5Product('monthly',
                                'pressure',
                                pressure_variables,
                                domain)
    era_pressure_files = pressure_product.download(t_0, t_1)

    # Interpolate pressure data.

    era5_data = xarray.open_dataset(era_pressure_files[0])
    lats_era = era5_data["latitude"][::-1]
    lons_era = era5_data["longitude"]
    p_era = era5_data["level"]
    p_inds = [np.where(p_era == p)[0] for p in [950, 850, 700, 500, 250]]
    pressures = []
    for ind in p_inds:
        p_interp = RegularGridInterpolator([lats_era, lons_era], era5_data["t"].data[0, ind[0], ::-1, :])
        pressures.append(p_interp((lats_r, lons_r)))


    era5_data = xarray.open_dataset(era_surface_files[0])
    lats_era = era5_data["latitude"][::-1]
    lons_era = era5_data["longitude"]
    t_interp = RegularGridInterpolator([lats_era, lons_era], era5_data["t2m"].data[0, ::-1, :])
    t_surf = t_interp((lats_r, lons_r))
    p_interp = RegularGridInterpolator([lats_era, lons_era], era5_data["sp"].data[0, ::-1, :])
    p_surf = p_interp((lats_r, lons_r))

    lats = lats_r[1:-1, 1:-1]
    lons = lons_r[1:-1, 1:-1]

    x = np.zeros(lats.shape + (16,))
    x[:, :, 0] = p_surf[1:-1, 1:-1]
    x[:, :, 1] = t_surf[1:-1, 1:-1]
    for i, p in enumerate(pressures):
        x[:, :, 2 + i] = p[1:-1, 1:-1]

    x[:, :, 8] = bt_12[1:-1, 1:-1]
    x[:, :, 9] = bt_11[1:-1, 1:-1] - bt_12[1:-1, 1:-1]
    x[:, :, 10] = bt_11_w - bt_12_w
    x[:, :, 11] = bt_11_c - bt_12_c
    x[:, :, 12] = bt_12_w - bt_12[1:-1, 1:-1]
    x[:, :, 13] = bt_12_c - bt_12[1:-1, 1:-1]

    x[:, :, 14] = bt_11_s
    x[:, :, 15] = bt_1112_s

    return lats, lons, x
