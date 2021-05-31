"""
quantnn.examples.ctp
====================

This module implements helper functions for the MODIS cloud-top pressure
example.
"""
from pathlib import Path
from urllib.request import urlretrieve

_DATA_PATH = "/home/simonpf/src/pansat/notebooks/products/data/"
_MODIS_FILES = [
    _DATA_PATH + "MODIS/MYD021KM.A2016286.1750.061.2018062214718.hdf",
    _DATA_PATH + "MODIS/MYD03.A2016286.1750.061.2018062032022.hdf",
]

modis_files = _MODIS_FILES
pad_along_orbit = 200
pad_across_orbit = 300


def prepare_input_data(modis_files):
    """
    Prepares validation data for the MODIS CTP retrieval.

    Args:
        modis_file: List of filenames containing the MODIS input data to use
            as input.

    Returns:
        Dictionary containing the different files required to run the retrieval
        on the CALIOP data.
    """
    from datetime import datetime
    from satpy import Scene
    import scipy as sp
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import maximum_filter, minimum_filter
    from scipy.signal import convolve
    import numpy as np
    import xarray
    from pansat.products.reanalysis.era5 import ERA5Product
    from pansat.products.satellite.calipso import clay01km
    from pykdtree.kdtree import KDTree
    from PIL import Image

    #
    # Prepare MODIS data.
    #

    scene = Scene(filenames=modis_files, reader="modis_l1b")
    scene.load(["true_color", "31", "32", "latitude", "longitude"], resolution=1000)

    scene["true_color_small"] = scene["true_color"][:, ::4, ::4]
    scene["bt_11_small"] = scene["31"][::4, ::4]
    scene["bt_12_small"] = scene["32"][::4, ::4]
    scene["latitude_small"] = scene["latitude"][::4, ::4]
    scene["longitude_small"] = scene["longitude"][::4, ::4]
    scene.save_dataset("true_color_small", "modis_true_color.png")
    image = Image.open("modis_true_color.png")
    modis_rgb = np.array(image)
    bt_11_rgb = scene["bt_11_small"].compute()
    bt_12_rgb = scene["bt_12_small"].compute()
    lats_rgb = scene["latitude_small"].compute()
    lons_rgb = scene["longitude_small"].compute()

    # MODIS data input features.

    lats_r = scene["latitude"].compute()
    lons_r = scene["longitude"].compute()
    bt_11 = scene["31"].compute()
    bt_12 = scene["32"].compute()

    def mean_filter(img):
        k = np.ones((5, 5)) / 25.0
        return convolve(img, k, mode="same")

    def std_filter(img):
        mu = mean_filter(img ** 2)
        mu2 = mean_filter(img) ** 2
        return np.sqrt(mu - mu2)

    bt_11_w = maximum_filter(bt_11, [5, 5])
    bt_11_c = minimum_filter(bt_11, [5, 5])
    bt_12_w = maximum_filter(bt_12, [5, 5])
    bt_12_c = minimum_filter(bt_12, [5, 5])
    bt_11_s = std_filter(bt_11)
    bt_1112_s = std_filter(bt_11 - bt_12)

    #
    # Calipso data
    #

    t_0 = datetime(2016, 10, 12, 17, 00)
    t_1 = datetime(2016, 10, 12, 17, 50)
    calipso_files = clay01km.download(t_0, t_1)

    lat_min = lats_r.data.min()
    lat_max = lats_r.data.max()
    lon_min = lons_r.data.min()
    lon_max = lons_r.data.max()

    dataset = clay01km.open(calipso_files[0])
    lats_c = dataset["latitude"].data
    lons_c = dataset["longitude"].data
    ctp_c = dataset["layer_top_pressure"]
    cth_c = dataset["layer_top_altitude"]

    indices = np.where(
        (lats_c > lat_min)
        * (lats_c <= lat_max)
        * (lons_c > lon_min)
        * (lons_c <= lon_max)
    )
    points = np.hstack([lats_r.data.reshape(-1, 1), lons_r.data.reshape(-1, 1)])
    kd_tree = KDTree(points)

    points_c = np.hstack([lats_c.reshape(-1, 1), lons_c.reshape(-1, 1)])
    d, indices = kd_tree.query(points_c)
    valid = d < 0.01
    indices = indices[valid]
    lats_c = lats_c[valid]
    lons_c = lons_c[valid]
    ctp_c = ctp_c[valid]
    cth_c = cth_c[valid]
    bt_11 = bt_11.data.ravel()[indices]
    bt_12 = bt_12.data.ravel()[indices]
    bt_11_w = bt_11_w.ravel()[indices]
    bt_12_w = bt_12_w.ravel()[indices]
    bt_11_c = bt_11_c.ravel()[indices]
    bt_12_c = bt_12_c.ravel()[indices]
    bt_11_s = bt_11_s.ravel()[indices]
    bt_1112_s = bt_1112_s.ravel()[indices]
    lats_r = lats_r.data.ravel()[indices]
    lons_r = lons_r.data.ravel()[indices]

    #
    # ERA 5 data.
    #

    t_0 = datetime(2016, 10, 12, 17, 45)
    t_1 = datetime(2016, 10, 12, 17, 50)

    surface_variables = ["surface_pressure", "2m_temperature", "tcwv"]
    domain = [lat_min - 2, lat_max + 2, lon_min - 2, lon_max + 2]
    surface_product = ERA5Product("hourly", "surface", surface_variables, domain)
    era_surface_files = surface_product.download(t_0, t_1)

    pressure_variables = ["temperature"]
    pressure_product = ERA5Product("hourly", "pressure", pressure_variables, domain)
    era_pressure_files = pressure_product.download(t_0, t_1)

    # interpolate pressure data.

    era5_data = xarray.open_dataset(era_pressure_files[0])
    lats_era = era5_data["latitude"][::-1]
    lons_era = era5_data["longitude"]
    p_era = era5_data["level"]
    p_inds = [np.where(p_era == p)[0] for p in [950, 850, 700, 500, 250]]
    pressures = []
    for ind in p_inds:
        p_interp = RegularGridInterpolator(
            [lats_era, lons_era], era5_data["t"].data[0, ind[0], ::-1, :]
        )
        pressures.append(p_interp((lats_r, lons_r)))

    era5_data = xarray.open_dataset(era_surface_files[0])
    lats_era = era5_data["latitude"][::-1]
    lons_era = era5_data["longitude"]
    t_interp = RegularGridInterpolator(
        [lats_era, lons_era], era5_data["t2m"].data[0, ::-1, :]
    )
    t_surf = t_interp((lats_r, lons_r))
    p_interp = RegularGridInterpolator(
        [lats_era, lons_era], era5_data["sp"].data[0, ::-1, :]
    )
    p_surf = p_interp((lats_r, lons_r))
    tcwv_interp = RegularGridInterpolator(
        [lats_era, lons_era], era5_data["tcwv"].data[0, ::-1, :]
    )
    tcwv = tcwv_interp((lats_r, lons_r))

    #
    # Assemble input data
    #

    x = np.zeros((lats_r.size, 16))
    x[:, 0] = p_surf
    x[:, 1] = t_surf
    for i, p in enumerate(pressures):
        x[:, 2 + i] = p
    x[:, 7] = tcwv

    x[:, 8] = bt_12
    x[:, 9] = bt_11 - bt_12
    x[:, 10] = bt_11_w - bt_12_w
    x[:, 11] = bt_11_c - bt_12_c
    x[:, 12] = bt_12_w - bt_12
    x[:, 13] = bt_12_c - bt_12

    x[:, 14] = bt_11_s
    x[:, 15] = bt_1112_s

    output_data = {
        "input_data": x,
        "ctp": ctp_c,
        "latitude": lats_r,
        "longitude": lons_r,
        "latitude_rgb": lats_rgb,
        "longitude_rgb": lons_rgb,
        "modis_rgb": modis_rgb,
        "bt_11_rgb": bt_11_rgb,
        "bt_12_rgb": bt_12_rgb,
    }
    return output_data


def download_data(destination="data"):
    """
    Downloads training and evaluation data for the CTP retrieval.

    Args:
        destination: Where to store the downloaded data.
    """
    datasets = ["ctp_training_data.npz", "ctp_validation_data.npz"]

    Path(destination).mkdir(exist_ok=True)

    for file in datasets:
        file_path = Path("data") / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/ctp/{file}"
            urlretrieve(url, file_path)
