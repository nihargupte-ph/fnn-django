import os
import geopandas as gpd
import xarray
from skimage.util.shape import view_as_windows
import numpy as np
import xarray
from pyproj import CRS
import rioxarray
import sys
import logging
import time

from detect_fires import config
from detect_fires.nn import nn7
from detect_fires import misc_functions

def predict_xds(xds_path, nn):
    """ 
    Parameters
    ----------
    xds_path : str
        path to the netcdf4 file we are trying to predict
    nn : fnn.nn.NN 
        neural network object which will be used to predict
        
    Returns  
    ----------
    pred_xds : xaray.Dataset
        predicted dataset
    """

    try:
        filepath_dict = misc_functions.closest_dates(xds_path, nn.prev_time_measurements, nn.important_bands)
    except:
        return 'wait'

    if len(filepath_dict[nn.band_of_interest]) != nn.prev_time_measurements:
        return 'wait'

    arr_lst = []
    for nc_file in filepath_dict[nn.band_of_interest]:
        xds = xarray.open_dataset(nc_file)
        # Clipping to California
        arr_lst.append(np.pad(xds.Rad.values, pad_width=2, mode='constant', constant_values=np.nan))
        xds.close()
    padded_arr = np.stack(arr_lst, axis=2)

    # We just need the properties, the long and lat will be completely different
    pred_xds = xds.copy()

    X = view_as_windows(padded_arr, (5, 5, 4)).reshape(xds.Rad.values.shape[0], xds.Rad.values.shape[1], 100)
    X = np.delete(X, 48, axis=2)  # deleting the middle index

    # Need to flatten to pass into NN
    shp = X.shape
    X_flat = X.reshape(shp[0]*shp[1], shp[2])
    x_trans = nn.scaler_x.transform(X_flat)
    prediction_trans = nn.model.predict(x_trans)
    prediction = nn.scaler_y.inverse_transform(prediction_trans)
    prediction = prediction.reshape(shp[0], shp[1])

    # Replacing xds.Rad with prediction
    pred_xds.Rad.values = prediction
    return pred_xds

def normalize_xds(xds, proj="EPSG:4326"):
    """ 
    Parameters
    ----------
    xds : xarray.Dataset
        dataset we want to normalize
    proj : str, optional
        WPSG projection which will be reprojected to. Defaults to lon lat coordinates

    Returns  
    ----------
    proj_xds : xarray.Dataset
        normalized xarray according to sattelitte height
    """

    #Try looking for perspective_point_height, if it doesn't exist then we already noramlized it
    try:
        sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
    except AttributeError:
        return xds
    # xds.assign_coords({"x": xds.x.values * sat_height, "y": xds.y.values * sat_height})
    xds.x.values *= sat_height
    xds.y.values *= sat_height
    cc = CRS.from_cf(xds.goes_imager_projection.attrs)
    xds.rio.write_crs(cc.to_string(), inplace=True)
    xds = xds[["Rad", "DQF"]]
    proj_xds = xds.rio.reproject(proj)

    return proj_xds

def clip_xds_cali(xds):
    """ 
    Parameters
    ----------
    xds : xarray.Dataset

    Returns  
    ----------
    clipped_xds : xarray.Dataset
        dataset clipped to only california
    """
    proj_xds = xds.rio.reproject("EPSG:4326")
    dirname = os.path.dirname(__file__)
    df = gpd.read_file(os.path.join(config.CALI_SHP_FOLDER, "CA_State_TIGER2016.shp"))
    df = df.to_crs("EPSG:4326")
    geometry = df.geometry[0]

    clipped_xds = proj_xds.rio.clip([geometry], "EPSG:4326")
    return clipped_xds

def predict_xarray(actual_xds_path):
    """ 
    Parameters
    ----------
    actual_xds_path : str
        path to the xarray.Dataset/netCDF4 file we are trying to predict. Could be band 14 or 7
    nn : fnn.nn.NN, optional
        if not specified will default to band 7. Note that this function should only
        be used to predict in band 7, the only reason there is an option of actually providing
        a NN is so that we can reduce the number of times NN is loaded

    Returns  
    ----------
    ret : str
        path to predicted xarray
    """
    logging.info('entered prediction')
    nn = nn7()
    try:
        pred_xds = predict_xds(actual_xds_path, nn)
        if not isinstance(pred_xds, xarray.Dataset):
            logging.warning(f'Not enough band 7 data to predict needs {nn.prev_time_measurements}')
            return None
        else:
            logging.info("Successfully predicted xarray")
    except:
        logging.critical(sys.exc_info()[0])
        raise Exception("Unable to predict xarray")

    basename = os.path.basename(actual_xds_path)
    diff_xds = pred_xds.copy()
    actual_xds = xarray.open_dataset(actual_xds_path)
    diff_xds.Rad.values = actual_xds.Rad.values - pred_xds.Rad.values
    logging.info("Successfully calculated difference xarray")

    pred_path = os.path.join(config.NC_DATA_FOLDER, "ABI_RadC", "pred", 'pred', basename)
    pred_xds.to_netcdf(path=pred_path)
    logging.info(f"Successfully saved predicted xarray at {pred_path}")
    diff_path = os.path.join(config.NC_DATA_FOLDER, "ABI_RadC", 'pred', "diff", basename)
    diff_xds.to_netcdf(path=diff_path)
    logging.info(f"Successfully saved difference xarray at {diff_path}")
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3)
    xds = xarray.open_dataset(actual_xds_path)
    pred_xds.Rad.plot(ax=axs[0])
    xds.Rad.plot(ax=axs[1])
    diff_xds.Rad.plot(ax=axs[2])
    plt.show()

    pred_xds.close()
    actual_xds.close()

    return actual_xds_path
        
def predict_cloud(actual_xds_path, cloud_value=1, num_before=8):
    """ 
    Parameters
    ----------
    actual_xds_path : str
        path to the xarray.Dataset/netCDF4 file we are trying to find the cloud array of. Should be 
        band 14

    Returns  
    ----------
    ret : str 
        path to cloud xarray 
    """
    filepaths = misc_functions.closest_dates(actual_xds_path, num_before, [14])[14]
    
    try:
        arr = np.stack([xarray.open_dataset(filename).Rad.values for filename in filepaths], axis=2)
    except:
        logging.warning("Only one band 14 file waiting for more...")
        return None

    if arr.shape[-1] != num_before:
        logging.warning(f'not enough data to predict clouds with {num_before} previous measurements, using {arr.shape[-1]} instead')
        num_before = arr.shape[-1]
    
    X = np.apply_along_axis(np.diff, 2, arr)
    X = np.apply_along_axis(np.abs, 2, X)
    # X = np.apply_along_axis(np.log, 2, X)
    def comparison(x): return x < cloud_value
    X = np.apply_along_axis(comparison, 2, X)

    X = np.apply_along_axis(np.all, 2, X)

    cloud_xds = xarray.open_dataset(filepaths[0]).copy()
    cloud_xds.Rad.values = np.full([cloud_xds.Rad.values.shape[0], cloud_xds.Rad.values.shape[1]], None)
    X = xarray.DataArray(X, dims={'y':cloud_xds.y.values, 'x':cloud_xds.x.values})
    cloud_xds['Cloud'] = X
    cloud_xds = cloud_xds.drop(['Rad'])

    basename = os.path.basename(actual_xds_path)
    cloud_path = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud', basename)
    try:
        cloud_xds.to_netcdf(path=cloud_path, mode='w')
        logging.info("Successfully saved cloud xarray")
    except:
        logging.error("Unable to save cloud xarray")

    return cloud_path

def classify(bandpath_dct):
    """ 
    Parameters
    ----------
    bandpath_dct : dict
        dictionary containing paths of predicted xarray and cloud xarray

    cloud_xds_path : str
        path to the xarray.Dataset/netCDF4 file which contains cloud truth array

    Returns  
    ----------
    new_fires_lst : list
        List of fires detected. Note will return empty if no new fires are detected. If new fires
        are detected then will return a list of form [(lon, lat, timestamp, location of fire in db)]
    """
    # Extacting filepaths from tuple
    diff_xds_path = bandpath_dct['diff']
    cloud_xds_path = bandpath_dct['cloud']

    diff_basename = os.path.basename(diff_xds_path)
    cloud_basename = os.path.basename(cloud_xds_path)
    diff_xds = xarray.open_dataset(diff_xds_path)
    cloud_xds = xarray.open_dataset(cloud_xds_path)

    cloud_conf, non_cloud_conf = .55, .17
    cloud_conf = np.full([diff_xds.Rad.values.shape[0], diff_xds.Rad.values.shape[1]], cloud_conf)
    non_cloud_conf = np.full([diff_xds.Rad.values.shape[0], diff_xds.Rad.values.shape[1]], non_cloud_conf)

    # Edges have nan values so we can't predict
    nan_truth_arr = np.full(shape=diff_xds.Rad.values.shape, fill_value=True, dtype=bool)
    nan_args = np.where(np.isnan(diff_xds.Rad.values))
    nan_truth_arr[nan_args] = False

    # Finding where the confidence interval is exceeded for clouded areas
    cloud_truth = diff_xds.Rad.values > cloud_conf
    cloud_truth = np.logical_and(cloud_truth, ~cloud_xds.Cloud.values)

    # Finding where the confidence interval is exceeded for non-clouded areas
    non_cloud_truth = diff_xds.Rad.values > non_cloud_conf
    non_cloud_truth = np.logical_and(non_cloud_truth, cloud_xds.Cloud.values)

    # Combining
    truth_arr = np.logical_or(cloud_truth, non_cloud_truth)
    truth_arr = np.logical_and(truth_arr, nan_truth_arr)

    # Getting lat lon time idxs
    lat_idxs, lon_idxs = np.where(truth_arr == True)
    anomaly_lats, anomaly_lons = diff_xds.y.values[lat_idxs], diff_xds.x.values[lon_idxs]
    anomaly_time = diff_xds.t.values

    print(len(lat_idxs))
    print(truth_arr.shape[0] * truth_arr.shape[1])

    diff_xds.close()
    cloud_xds.close()

    # Deleting files now that we have used one "group" of data
    # closest_band14_file = misc_functions.find_closest_file(basename, os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_14'))

    # classify("/home/n/Documents/Research/fnn/media/ABI_RadC/pred/diff/OR_ABI-L1b-RadC-M6C07_G16_s20202250521181_e20202250523566_c20202250524075.nc")

    # # Removing oldest file in actual, prediction, diff, and cloud array if we have at least 12 files (only band 7)
    # max_files = 12
    # folder_lst = [os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_7'), os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff'), os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'pred'), os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud')]

    # # If every folder has more than enough files
    # try:
    #     if (len(os.listdir(folder_lst[0])) > max_files) and (len(os.listdir(folder_lst[1])) > max_files) and (len(os.listdir(folder_lst[2])) > max_files) and (len(os.listdir(folder_lst[3])) > max_files):
    #         for folder in folder_lst:
    #             os.remove(misc_functions.find_oldest_file())
    #         logging.info(f"Successfully deleted oldest files from {folder_lst}")
    # except:
    #     logging.error("Unable to delete files" + str(sys.exc_info()[0]))


# import matplotlib.pyplot as plt
# dct = {'diff': '/home/n/Documents/Research/fnn/media/ABI_RadC/pred/diff/OR_ABI-L1b-RadC-M6C07_G16_s20202322116183_e20202322118567_c20202322119084.nc', 'cloud': '/home/n/Documents/Research/fnn/media/ABI_RadC/pred/cloud/OR_ABI-L1b-RadC-M6C14_G16_s20202322116183_e20202322118556_c20202322119072.nc'}
# classify(dct)