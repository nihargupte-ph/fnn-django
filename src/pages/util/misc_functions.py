import numpy as np
import os
import datetime
import time
import pickle
import logging
import base64
import matplotlib.pyplot as plt
import matplotlib
import string
import random
import io
import pytz

import xarray
from pyproj import CRS
import rioxarray
from django.core.files.base import ContentFile

from pages.util import config

def xy_to_idx(x_coord, y_coord, xds, outproj="EPSG:4326", inproj="EPSG:4326"):
    """ Given a x coord and y coord value will find the in index of the xds which that xy coord pix is in. Defaults to longitude lattitude values """
    from pyproj import Proj, transform
    import numpy as np


    inproj = Proj(inproj)
    outproj = Proj(outproj)
    lat_lon = transform(inproj, outproj, y_coord, x_coord)
    lat, lon = lat_lon

    tmp_xds = xds.sel(x=lon, y=lat, method='nearest')
    x, y = tmp_xds.x.values, tmp_xds.y.values
    idx_x = np.where(xds.x.values == x)[0][0]
    idx_y = np.where(xds.y.values == y)[0][0]

    return (idx_x, idx_y)

def mini_normalize(xds):
    """ 
    Parameters
    ----------
    xds : xarray.Dataset
        Dataset we want to mini normalize

    Return
    ----------
    xds : xarray.Dataset
        mini normalized xarray.Dataset
    """
    cc = CRS.from_cf(xds.goes_imager_projection.attrs)
    xds.rio.write_crs(cc.to_string(), inplace=True)
    return xds

def dt64_to_timestamp(dt):
    """ Parameters
        ----------
        dt : np.datetime64 object
        
        Returns  
        ----------
        ret : float
            timestamp

        """
    ret = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return ret 

def dt64_to_datetime(dt):
    """ Parameters
    ----------
    dt : np.datetime64 object
    
    Returns  
    ----------
    ret : datetime.datetime
        converted datetime.datetime
    """
    ret = datetime.datetime.utcfromtimestamp(dt64_to_timestamp(dt)).replace(tzinfo=pytz.utc)
    return ret

def closest_dates(nc_filepath, prev_time_measurements, important_bands):
    """ 
    Parameters
    ----------
    nc_filepath : str
        path to the netcdf4 file we are trying to predict
    prev_time_measurements : int
        how many measurements before to use to predict
    important_bands : list
        the bands which are used to predict

    Returns  
    ----------
    filepath_dict : dict
        dictionary with keys as bands and values as an ordered list of 
        filepaths. Noe that it will also include the original provided 
        filepath.
    """
    nc_filename = os.path.split(nc_filepath)[1]
    nc_folder = os.path.split(nc_filepath)[0]

    mega_direc = os.listdir(nc_folder)

    wtp_band, wtp_tstart, wtp_tend = parse_goes16_file(nc_filename)

    parse_label_lst = [parse_goes16_file(f) for f in mega_direc]
    sorted_label_lst = sorted(parse_label_lst, key=lambda x: x[1])
    wtp_sort_idx = sorted_label_lst.index((wtp_band, wtp_tstart, wtp_tend)) + 1 #To include current time as well

    band_file_idxs = {i:[] for i in important_bands} #Dict of lists containing indexes of files prev_time_measurements many measurements 
    for important_band in important_bands:
        for tup in sorted_label_lst[wtp_sort_idx:0:-1]:
            if tup[0] == important_band:
                band_file_idxs[important_band].append(parse_label_lst.index(tup)) #Appending index in file not index in sorted datetime list
            if len(band_file_idxs[important_band]) == prev_time_measurements: #If we get enough measurements
                break
    
    filepath_dict = {i:[] for i in important_bands}
    for important_band in important_bands:
        for idx in band_file_idxs[important_band]:
            filepath_dict[important_band].append(f"{nc_folder}/{mega_direc[idx]}")

    return filepath_dict

def parse_goes16_file(file_string):
    """ 
    Parameters
    ----------
    file_string : str
        file we want to get the data from

    Returns  
    ----------
    (band_id, t_start, t_end) : tuple
        band_id, start, and end of netcdf4 file
    """

    """ Given file string of goes 16 file, will parse and returns band_id, start, and end  """
    # Search for the GOES-R channel in the file name
    try:
        band_id = int((file_string[file_string.find("_G16") - 2 :file_string.find("_G16")]))
    except:
        band_id = None

    
    # Search for the Scan Start in the file name
    t_start = (file_string[file_string.find("s")+1:file_string.find("_e")])
    t_start = datetime.datetime.strptime(t_start, "%Y%j%H%M%S%f")

    # Search for the Scan End in the file name
    t_end = (file_string[file_string.find("e")+1:file_string.find("_c")])
    t_end = datetime.datetime.strptime(t_end, "%Y%j%H%M%S%f")

    return (band_id, t_start, t_end)

def find_oldest_file(folder):
    """ 
    Parameters
    ----------
    folder : str
        folder we are looking inside of to get the oldest dated file

    Returns  
    ----------
    filepath : str
        filepath of oldest file
    """
    file_pair = [(key_from_filestring(filename), filename) for filename in os.listdir(folder)]
    file_pair = sorted(file_pair, key=lambda x: x[0])
    return file_pair[1][1]

def time_of_file(file_string):
    """ 
    Parameters
    ----------
    file_string : str
        filename or filepath formatted in ABI format which we are getting the time of 

    Returns  
    ----------
    dt : datetime.datetime
        datetime of average of start and end time of the file
    """
    filename = os.path.basename(file_string)
    tup = parse_goes16_file(filename)
    time = tup[1] + ((tup[2] - tup[1]) / 2)

    return time

def key_from_filestring(file_string):
    """ 
    Parameters
    ----------
    file_string : str
        filepath or filename we want the unique key of

    Returns  
    ----------
    key : int
    """
    file_string = os.path.basename(file_string)
    key = ((file_string[file_string.find("s")+1:file_string.find("_e")]))
    return key

def binfield_to_obj(bin_field):
    """ 
    Parameters
    ----------
    bin_field : models.BinaryField
        django binary field 

    Returns  
    ----------
    obj : obj
        numpy array or list created from django binary field
    """
    np_bytes = base64.b64decode(bin_field)
    obj = pickle.loads(np_bytes)
    return obj

def obj_to_binfield(obj):
    """ 
    Parameters
    ----------
    obj : obj
        numpy array or list

    Returns  
    ----------
    bin_field : models.BinaryField
        Django binary field
    """
    np_bytes = pickle.dumps(obj)
    np_base64 = base64.b64encode(np_bytes)
    
    return np_base64

def geodesic_point_buffer(lon, lat, km):

    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    from shapely.ops import transform
    from functools import partial

    # Azimuthal equidistant projection
    aeqd_proj = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0"
    transformer = Transformer.from_crs("EPSG:4326", aeqd_proj, always_xy=True)
    point = Point(lon, lat)
    point_aeqd = transform(transformer.transform, point)
    circle_aeqd = point_aeqd.buffer(km * 1000)
    return mapping(transform(partial(transformer.transform, direction="INVERSE"), circle_aeqd))

def snap_picture(xds, lon, lat, radius):
    """ 
    Parameters
    ----------
    xds : xarray.Dataset
        Dataset we want to take a picture of 
    lon : float
        Longitude of the fire
    lat : float
        Latitude of the fire
    radius : float
        radius in km of image
    Returns
    ----------
    data : django.core.files.base.ContentFile
        Content file we will be saving into the django ImageField
    """

    matplotlib.use('Agg')

    xds = xds.rio.reproject("EPSG:3857")
    buffered_point = geodesic_point_buffer(lon=lon, lat=lat, km=radius)
    clipped_xds = xds.rio.clip([buffered_point], "EPSG:4326")
    fig, ax = plt.subplots()
    im = clipped_xds.Rad.plot(ax=ax, cmap=plt.cm.viridis, cbar_kwargs={'label': 'Alarm Level'})

    # Formatting plot
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("Zoomed in fire pictures")

    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    data = ContentFile(pic_IObytes.read(), name='temp.jpg')
    plt.close()
    xds.close()

    return data

def snap_video(lon, lat, radius):
    """ 
    Parameters
    ----------
    lon : float
        Longitude of the fire
    lat : float
        Latitude of the fire
    radius : float
        radius in km of image
    Returns
    ----------
    data : django.core.files.base.ContentFile
        Content file we will be saving into the django ImageField. This will be a 
        gif. 
    """
    from wand.image import Image

    # Getting ordered diff folder so far to create images
    diff_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff')
    diff_file_lst = os.listdir(diff_folder)
    diff_file_lst = sorted(diff_file_lst, key=key_from_filestring)

    image_data_lst = []
    for i, filename in enumerate(diff_file_lst):
        # Open File
        diff_xds = xarray.open_dataset(os.path.join(diff_folder, filename))

        # Project and Clip
        diff_xds = diff_xds.rio.reproject("EPSG:3857")
        buffered_point = geodesic_point_buffer(lon=lon, lat=lat, km=radius)
        clipped_xds = diff_xds.rio.clip([buffered_point], "EPSG:4326")
        fig, ax = plt.subplots()
        clipped_xds.Rad.plot(ax=ax, cmap=plt.cm.viridis, cbar_kwargs={'label': 'Alarm Level'})

        # Formatting plot
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("Zoomed in fire pictures")
        diff_xds.close()

        # Saving plot
        image_data = io.BytesIO()
        plt.savefig(image_data, format='png')
        image_data.seek(0)
        image_data_lst.append(image_data)
        with Image(file=image_data) as img:
            img.save(filename=f'/home/n/Documents/Research/fnn-django/src/media/tmp/hi{i}.png')
        plt.close()

    # Creating gif
    with Image() as gif:
        # Add new frames into sequence
        for image_data in image_data_lst:
            print(image_data)
            with Image(filename=image_data) as img:
                gif.sequence.append(img)
        # Create progressive delay for each frame
        for cursor in range(len(gif.sequence)):
            with gif.sequence[cursor] as frame:
                frame.delay = 10
    # Set layer type
    gif.type = 'optimize'

    gif.save(filename='/home/n/Documents/Research/fnn-django/src/media/tmp/animated.gif')
    # byte_gif = io.BytesIO(gif)

    return None

def get_graph_points(px_idx):
    """ 
    Parameters
    ----------
    px_idx : tuple
        tule of fire index which shows where in the diff_xds the 
        fire is 

    Returns
    ----------
    time_graph_pts, pred_graph_pts, diff_graph_pts, cloud_graph_pts, actual_7_graph_pts, actual_14_graph_pts : list
        list of points of current files in actual and pred folders
    """
    # Getting filepaths of previous points, everyline for readability
    pred_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'pred')
    pred_file_lst = os.listdir(pred_folder)
    pred_file_lst = sorted(pred_file_lst, key=key_from_filestring)
    prev_pts = len(pred_file_lst)

    diff_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff')
    diff_file_lst = os.listdir(diff_folder)
    diff_file_lst = sorted(diff_file_lst, key=key_from_filestring)
    diff_file_lst = diff_file_lst[-prev_pts:]

    cloud_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud')
    cloud_file_lst = os.listdir(cloud_folder)
    cloud_file_lst = sorted(cloud_file_lst, key=key_from_filestring)
    cloud_file_lst = cloud_file_lst[-prev_pts:]
    
    actual_7_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_7')
    actual_7_file_lst = os.listdir(actual_7_folder)
    actual_7_file_lst = sorted(actual_7_file_lst, key=key_from_filestring)
    actual_7_file_lst = actual_7_file_lst[-prev_pts:]

    actual_14_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_14')
    actual_14_file_lst = os.listdir(actual_14_folder)
    actual_14_file_lst = sorted(actual_14_file_lst, key=key_from_filestring)
    actual_14_file_lst = actual_14_file_lst[-prev_pts:]

    time_graph_pts, pred_graph_pts = [], []
    for xds_path in pred_file_lst:
        xds = xarray.open_dataset(os.path.join(pred_folder, xds_path))
        pred_graph_pts.append(xds.Rad.values[px_idx[1], px_idx[0]])
        time_graph_pts.append(dt64_to_datetime(xds.t.values))
        xds.close()

    diff_graph_pts = []
    for xds_path in diff_file_lst:
        xds = xarray.open_dataset(os.path.join(diff_folder, xds_path))
        diff_graph_pts.append(xds.Rad.values[px_idx[1], px_idx[0]])
        xds.close()

    cloud_graph_pts = []
    for xds_path in cloud_file_lst:
        xds = xarray.open_dataset(os.path.join(cloud_folder, xds_path))
        cloud_graph_pts.append(xds.Cloud.values[px_idx[1], px_idx[0]])
        xds.close()

    actual_7_graph_pts = []
    for xds_path in actual_7_file_lst:
        xds = xarray.open_dataset(os.path.join(actual_7_folder, xds_path))
        actual_7_graph_pts.append(xds.Rad.values[px_idx[1], px_idx[0]])
        xds.close()

    actual_14_graph_pts = []
    for xds_path in actual_14_file_lst:
        xds = xarray.open_dataset(os.path.join(actual_14_folder, xds_path))
        actual_14_graph_pts.append(xds.Rad.values[px_idx[1], px_idx[0]])
        xds.close()

    return time_graph_pts, pred_graph_pts, diff_graph_pts, cloud_graph_pts, actual_7_graph_pts, actual_14_graph_pts

def cluster_to_FireModel(cluster, diff_xds_path):
    """ 
    Parameters
    ----------
    cluster : np.array
        numpy array containing a list of anomalies of form (lon, lat, timestamp)
        will add the anomalies to a newly created FireModel
    diff_xds_path : str
        Path to diff_xds that we just predicted

    Returns
    ----------
    fire : pages.FireModel
        updated fire
    """
    from pages.models import FireModel
    from django.core.management import call_command

    xds = xarray.open_dataset(diff_xds_path)
    
    # Initial lon and lat is the place with the highest radiance value
    px_idx_lst = [xy_to_idx(lon, lat, xds) for lon, lat in zip(cluster[:, 0], cluster[:, 1])]
    vals = [xds.Rad.values[px_idx[1], px_idx[0]] for px_idx in px_idx_lst]
    ll_idx = vals.index(max(vals))
    lon = cluster[ll_idx, 0]
    lat = cluster[ll_idx, 1]
    # Intial timestamp is average of first cluster's timestamp
    avg_timestamp = datetime.datetime.utcfromtimestamp(np.mean([dt.timestamp() for dt in cluster[:, 2]])).replace(tzinfo=pytz.utc)

    # Anomaly list
    anom_arr_bin = obj_to_binfield(cluster.flatten())

    # Cluster list, we wrap in a list as when more clusters are added we want to append them to the array
    cluster_lst_bin = obj_to_binfield([cluster])

    # Snapping Video
    # content_file = snap_video(lon, lat, 20)

    # Snapping Pictures
    content_file = snap_picture(xds, lon, lat, 20)

    # Getting index in xarrays
    px_idx = px_idx_lst[ll_idx]

    # Getting graph points
    time_graph_pts, pred_graph_pts, diff_graph_pts, cloud_graph_pts, actual_7_graph_pts, actual_14_graph_pts = get_graph_points(px_idx)

    # TODO Insert Causes here

    # TODO Insert names, short_description, long_description, probability here
    
    fire = FireModel.objects.create(
        longitude=lon,
        latitude=lat,
        timestamp=avg_timestamp,
        latest_timestamp=avg_timestamp, 
        anomaly_arr=anom_arr_bin,
        cluster_lst=cluster_lst_bin,
        image=content_file, 
        px_idx_x=px_idx[0],
        px_idx_y=px_idx[1],
        time_graph_pts=obj_to_binfield(time_graph_pts),
        pred_graph_pts=obj_to_binfield(pred_graph_pts),
        diff_graph_pts=obj_to_binfield(diff_graph_pts),
        cloud_graph_pts=obj_to_binfield(cloud_graph_pts),
        actual_7_graph_pts=obj_to_binfield(actual_7_graph_pts),
        actual_14_graph_pts=obj_to_binfield(actual_14_graph_pts),
    )

    xds.close()

    # Sending emails
    try:
        call_command('send_emails', lon, lat, avg_timestamp, 'link')
        logging.info('Sent emails')
    except:
        logging.warning('Failed to send emails')

    return fire

def update_FireModel(cluster, fire):
    """ 
    Parameters
    ----------
    cluster : np.array
        numpy array containing a list of anomalies FireModel
    fire : pages.models.FireModel
        firemodel we are updating

    Returns
    ----------
    fire : pages.FireModel
        updated fire
    """
    # Add new cluster to end of cluster list
    current_cluster_lst = binfield_to_obj(fire.cluster_lst)
    current_cluster_lst.append(cluster)    

    # Add current anomaly list to old anomaly list
    current_anomaly_arr = binfield_to_obj(fire.anomaly_arr)
    current_anomaly_arr = np.append(current_anomaly_arr, cluster.flatten(), 0)
    avg_timestamp = datetime.datetime.utcfromtimestamp(np.mean([dt.timestamp() for dt in cluster[:, 2]])).replace(tzinfo=pytz.utc)
    
    # Update fire model
    fire.cluster_lst = obj_to_binfield(current_cluster_lst)
    fire.anomaly_arr = obj_to_binfield(current_anomaly_arr)
    fire.latest_timestamp = avg_timestamp
    
    fire.save()

    return fire

def update_FireModel_plots(diff_xds_path, cloud_xds_path, fire):
    """ 
    Parameters
    ----------
    diff_xds_path : str
        File that are using to update the plots, note that we will have to use multiple files, but the basename
        is the same 
    cloud_xds_path : str
        File that are using to update the plots, note that we will have to use multiple files, but the basename
        is the same 
    fire : pages.models.FireModel
        firemodel we are updating
    """
    diff_basname = os.path.basename(diff_xds_path)
    cloud_basename = os.path.basename(cloud_xds_path)

    # Getting old points
    time_graph_pts       = binfield_to_obj(fire.time_graph_pts)
    pred_graph_pts       = binfield_to_obj(fire.pred_graph_pts)
    diff_graph_pts       = binfield_to_obj(fire.diff_graph_pts)
    cloud_graph_pts      = binfield_to_obj(fire.cloud_graph_pts)
    actual_7_graph_pts   = binfield_to_obj(fire.actual_7_graph_pts)
    actual_14_graph_pts  = binfield_to_obj(fire.actual_14_graph_pts)
    px_idx_x             = fire.px_idx_x
    px_idx_y             = fire.px_idx_y

    # Opening new xds
    actual_7_xds = xarray.open_dataset(os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_7', diff_basname))
    actual_14_xds = xarray.open_dataset(os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_14', cloud_basename))
    pred_xds = xarray.open_dataset(os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'pred', diff_basname))
    diff_xds = xarray.open_dataset(os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff', diff_basname))
    cloud_xds = xarray.open_dataset(os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud', cloud_basename))

    # Getting new points
    new_time_pt = dt64_to_datetime(diff_xds.t.values)
    new_pred_pt = pred_xds.Rad.values[px_idx_y, px_idx_x]
    new_diff_pt = diff_xds.Rad.values[px_idx_y, px_idx_x]
    new_cloud_pt = cloud_xds.Cloud.values[px_idx_y, px_idx_x]
    new_actual_7_pt = actual_7_xds.Rad.values[px_idx_y, px_idx_x]
    new_actual_14_pt = actual_14_xds.Rad.values[px_idx_y, px_idx_x]

    # Combining
    time_graph_pts.append(new_time_pt)
    pred_graph_pts.append(new_pred_pt)
    diff_graph_pts.append(new_diff_pt)
    cloud_graph_pts.append(new_cloud_pt)
    actual_7_graph_pts.append(new_actual_7_pt)
    actual_14_graph_pts.append(new_actual_14_pt)

    # Updating fire model 
    fire.time_graph_pts = obj_to_binfield(time_graph_pts)
    fire.pred_graph_pts = obj_to_binfield(pred_graph_pts)
    fire.diff_graph_pts = obj_to_binfield(diff_graph_pts)
    fire.cloud_graph_pts = obj_to_binfield(cloud_graph_pts)
    fire.actual_7_graph_pts = obj_to_binfield(actual_7_graph_pts)
    fire.actual_14_graph_pts = obj_to_binfield(actual_14_graph_pts)

    fire.save()

    actual_7_xds.close()
    actual_14_xds.close()
    pred_xds.close()
    diff_xds.close()
    cloud_xds.close()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def unnan_arr(data):
    """ 
    Parameters 
    ------------
    data : np.array
        numpy array we want to remove nans from
    data : np.array
        cleaned numpy array linearly inerpolated where there are nans
    """

    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

    return data