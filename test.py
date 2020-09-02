import xarray
import os
import numpy as np
import datetime
import pytz

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

def update_lightning(fire):
    """ 
    Parameters
    ----------
    fire : pages.models.FireModel
        Fire model we are checking to see if there is lightning data on

    Returns  
    ----------
    fire : pages.models.FireModel
        Updated fire model. If there is or is not lightning
    """

    # Combining xarray datasets in the folder this allows us to only worry about spatial search
    flash_lat_lst, flash_lon_lst, flash_time_lst = [], [], []
    glm_folder = os.path.join('/home/n/Documents/Research/fnn-django/src/media/data/GLM')
    for xds_name in os.listdir(glm_folder):
        xds = xarray.open_dataset(os.path.join(glm_folder, xds_name))
        flash_lon_lst.append(xds.flash_lon.values)
        flash_lat_lst.append(xds.flash_lat.values)
        flash_time_lst.append(xds.flash_time_offset_of_first_event.values)

    flash_lon_arr = np.concatenate(flash_lon_lst)
    flash_lat_arr = np.concatenate(flash_lat_lst)
    flash_time_arr = np.concatenate(flash_time_lst)
    flash_time_arr = np.array([misc_functions.dt64_to_datetime(dt64) for dt64 in flash_time_arr])

    fire_lon = fire.lon
    fire_lat = fire.lat

    distance_arr = np.array([misc_functions.haversine(fire_lon, fire_lat, flash_lon, flash_lat) for flash_lon, flash_lat in zip(flash_lon_arr, flash_lat_arr)])

    if np.argmin(distance_arr) < 10: # km 
        closest_flash_idx = np.argmin(distance_arr)
        flash = (flash_lon_arr[closest_flash_idx], flash_lat_arr[closest_flash_idx], flash_time_arr[closest_flash_idx])
        fire.lightning_lon = flash[0]
        fire.lightning_lat = flash[1]
        fire.lightning_timestamp = flash[2]

    return fire


fire = (38, -122, datetime.datetime(year=2020, month=9, day=1))
update_lightning(fire)