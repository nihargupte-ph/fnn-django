import os
import datetime
import numpy as np
import sys
import logging
import time
import pickle

import geopy.distance
import geopandas as gpd
from skimage.util.shape import view_as_windows
from pyproj import CRS
import rioxarray
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import xarray
from django.core.management import call_command
from django.core import management

from pages.util import config 
from pages.util import misc_functions
from pages.util.nn import nn7
from pages.models import FireModel

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
        path to difference array
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
    actual_xds = xarray.open_dataset(actual_xds_path)
    
    # I would have liked to create a copy with actual_xds.copy() and change only the Rad value. However, when saving with to_netcdf the fire 
    # miraculously changes so for now this is the workaround. A manuel copy. NOTE t is in coords as I couldn't figure out how to make a proper
    # datavar
    diff_xds = xarray.Dataset(
            data_vars={
                "Rad": (("y", "x"), actual_xds.Rad.values - pred_xds.Rad.values),
                "DQF": (("y", "x"), actual_xds.DQF.values - pred_xds.DQF.values),
                },
            coords={
                "x": actual_xds.x,
                "y": actual_xds.y,
                "x_image": actual_xds.x_image,
                "y_image": actual_xds.y_image,
                "t": actual_xds.t,
                "goes_imager_projection": actual_xds.goes_imager_projection
            },
            attrs={
                "naming_authority": actual_xds.naming_authority,
                "Conventions": actual_xds.Conventions,
                "standard_name_vocabulary": actual_xds.standard_name_vocabulary,
                "institution": actual_xds.institution,
                "project": actual_xds.project,
                "production_site": actual_xds.production_site,
                "production_environment": actual_xds.production_environment,
                "spatial_resolution": actual_xds.spatial_resolution,
                "Metadata_Conventions": actual_xds.Metadata_Conventions,
                "orbital_slot": actual_xds.orbital_slot,
                "platform_ID": actual_xds.platform_ID,
                "instrument_type": actual_xds.instrument_type,
                "scene_id": actual_xds.scene_id,
                "instrument_ID": actual_xds.instrument_ID,
                "title": actual_xds.title, 
                "summary": actual_xds.summary,
                "keywords": actual_xds.keywords,
                "keywords_vocabulary": actual_xds.keywords_vocabulary,
                "iso_series_metadata_id": actual_xds.iso_series_metadata_id,
                "license": actual_xds.license,
                "processing_level": actual_xds.processing_level,
                "cdm_data_type": actual_xds.cdm_data_type,
                "dataset_name": actual_xds.dataset_name,
                "production_data_source": actual_xds.production_data_source,
                "timeline_id": actual_xds.timeline_id,
                "date_created": actual_xds.date_created,
                "time_coverage_start": actual_xds.time_coverage_start, 
                "time_coverage_end": actual_xds.time_coverage_end,
                "LUT_Filenames": actual_xds.LUT_Filenames,
                "id": actual_xds.id,
                "grid_mapping": actual_xds.grid_mapping
            }
        )
    # Giving a CRS back to the array
    diff_xds = misc_functions.mini_normalize(diff_xds)

    logging.info("Successfully calculated difference xarray")

    pred_path = os.path.join(config.NC_DATA_FOLDER, "ABI_RadC", "pred", 'pred', basename)
    if os.path.exists(pred_path):
        os.remove(pred_path)
    pred_xds.to_netcdf(path=pred_path)
    logging.info(f"Successfully saved predicted xarray at {pred_path}")
    diff_path = os.path.join(config.NC_DATA_FOLDER, "ABI_RadC", 'pred', "diff", basename)
    if os.path.exists(diff_path):
        os.remove(diff_path)
    diff_xds.to_netcdf(path=diff_path)
    logging.info(f"Successfully saved difference xarray at {diff_path}")

    
    actual_xds.close()

    return diff_path
        
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
    """

    def get_anomalies(bandpath_dct):
        """ 
        Parameters
        ----------
        bandpath_dct : dict
            dictionary containing paths of predicted xarray and cloud xarray

        Returns  
        ----------
        ret : list
            List of form (lon, lat, datetime.datetime) of anomalies detected
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

        diff_xds.close()
        cloud_xds.close()

        dt = datetime.datetime.utcfromtimestamp(misc_functions.dt64_to_timestamp(anomaly_time))
        ret = list(zip(anomaly_lons, anomaly_lats, [dt for _ in anomaly_lats]))
        return ret


    def cluster_anomalies(anomaly_lst):
        """ 
        Parameters
        ----------
        anomaly_lst : list
            list of anomalies of form (lon, lat)

        Returns  
        ----------
        cluster_lst : list
            list of np.arrays which were clustered with DBSCAN 
        """

        def km_distance_thresh(x, y):
            """ Returns 0 if within threshold and 1 if outside """
            dist = geopy.distance.distance((x[1], x[0]), (y[1], y[0])).km
            if dist < 10:  # km
                return .1
            else:
                return 1

        anomaly_arr = np.array(anomaly_lst)
        db = DBSCAN(eps=.11, metric=km_distance_thresh, min_samples=2).fit(anomaly_arr)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        unique_labels = set(db.labels_)

        cluster_lst = []
        for k in unique_labels:
            cluster_idxs = np.where(db.labels_ == k)
            cluster = anomaly_arr[cluster_idxs]
            cluster_lst.append(cluster)

        return cluster_lst


    logging.info('Entered classification')
    try:
        anomaly_lst = get_anomalies(bandpath_dct)
        logging.info(f"Got anomalies. {len(anomaly_lst)} anomalies detected")
    except:
        logging.critical("Unable to get anomalies")
    anomaly_time = anomaly_lst[0][-1]

    space_anomaly_lst = [(i[0], i[1]) for i in anomaly_lst]
    try: 
        cluster_lst = cluster_anomalies(space_anomaly_lst)
        logging.info(f"Clustered anomalies. {len(cluster_lst)} clusters formed")
    except:
        logging.critical("Unable to cluster anomalies")
    cluster_lst = [np.hstack((cluster, np.full(shape=(cluster.shape[0], 1), fill_value=anomaly_time))) for cluster in cluster_lst]

    found_center_lst = []
    for cluster in cluster_lst:
        found_lon = np.mean([i[0] for i in cluster])
        found_lat = np.mean([i[1] for i in cluster])
        found_center_lst.append((np.deg2rad(found_lat), np.deg2rad(found_lon)))
    
    time_filter = anomaly_time - datetime.timedelta(hours=1)
    queried_fires = FireModel.objects.filter(latest_timestamp__gt=time_filter)

    queried_center_lst = []
    for fire in queried_fires:
        query_cluster_lst = misc_functions.binfield_to_obj(fire.cluster_lst)
        recent_cluster = query_cluster_lst[-1] # NOTE maybe should be changed to previous 2 clusters but left as 1 for now
        recent_lon = np.mean([i[0] for i in recent_cluster])
        recent_lat = np.mean([i[1] for i in recent_cluster])
        queried_center_lst.append((np.deg2rad(recent_lat), np.deg2rad(recent_lon)))

    # Use ball tree to link the found cluster centers w/ the recent cluster centers
    if not len(queried_center_lst) == 0:
        tree = BallTree(np.array(queried_center_lst), leaf_size=2, metric='haversine')
        dist, index = tree.query(np.array(found_center_lst), k=1)
        logging.info('Constructed BallTree for linking old fires')

        for i, (ind,d) in enumerate(zip(index, dist)):
            d *= 6371 # convert to km
            cluster = cluster_lst[i]
            if d < 10:
                fire = queried_fires[int(ind)]
                misc_functions.update_FireModel(cluster, fire)
                logging.info("Updated fire")
            else:
                misc_functions.cluster_to_FireModel(cluster, bandpath_dct['diff'])
                logging.info("Created new fire")
    else:
        for cluster in cluster_lst:
            misc_functions.cluster_to_FireModel(cluster, bandpath_dct['diff'])
            logging.info("Created new fire")

    # Writing that we predicted using the file in classified_lst.pkl
    with open(os.path.join(config.MEDIA_FOLDER, 'misc', 'classified_lst.pkl'), 'rb') as f:
        classified_lst = pickle.load(f)
    # If the list is too long we want to clear the first 50 or so entries
    if len(classified_lst) > 100:
        classified_lst = classified_lst[:50]
    classified_lst.append(os.path.basename(bandpath_dct['diff']))
    with open(os.path.join(config.MEDIA_FOLDER, 'misc', 'classified_lst.pkl'), 'wb') as f:
        pickle.dump(classified_lst, f)

    # Deleting files now that we have used one "group" of data
    # Removing oldest file in actual, prediction, diff, and cloud array if we have at least 12 files (only band 7)
    max_files = 12
    folder_lst = [
        os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_7'), 
        os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_14'), 
        os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff'), 
        os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'pred'), 
        os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud'),
        ]

    # If every folder has more than enough files
    try:
        if all(len(folder) == max_files for folder in folder_lst):
            for folder in folder_lst:
                os.remove(misc_functions.find_oldest_file(folder))
            logging.info(f"Successfully deleted oldest files from {folder_lst}")
    except:
        logging.error("Unable to delete files\n" + str(sys.exc_info()[0]))


# TMP
# import xarray
# import matplotlib.pyplot as plt
# xds = xarray.open_dataset('/home/n/Documents/Research/fnn-django/src/media/data/ABI_RadC/pred/diff/OR_ABI-L1b-RadC-M6C07_G16_s20202391911173_e20202391913557_c20202391914083.nc')
# xds.Rad.plot()