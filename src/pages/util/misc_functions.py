import numpy as np
import os
import datetime
import time
import pickle
import logging
import base64

from pages.util import config

def dt64_to_timestamp(dt):
    """ Parameters
        ----------
        dt : np.datetime64 object
        
        Returns  
        ----------
        ret : string

        """
    ret = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
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

def cluster_to_FireModel(cluster):
    """ 
    Parameters
    ----------
    cluster : np.array
        numpy array containing a list of anomalies of form (lon, lat, timestamp)
        will add the anomalies to a newly created FireModel
    """
    from pages.models import FireModel

    # Intial lon, lat, and timestamp is average of first cluster
    avg_lat = np.mean(cluster[:, 0])
    avg_lon = np.mean(cluster[:, 1])
    avg_timestamp = datetime.datetime.utcfromtimestamp(np.mean([dt.timestamp() for dt in cluster[:, 2]]))

    # Anomaly list
    anom_arr_bin = obj_to_binfield(cluster.flatten())

    # Cluster list, we wrap in a list as when more clusters are added we want to append them to the array
    cluster_lst_bin = obj_to_binfield([cluster])

    # TODO Insert Picture taking here

    # TODO Insert Causes here

    # TODO Insert names, short_description, long_description, probability here
    
    FireModel.objects.create(
        longitude=avg_lon,
        latitude=avg_lat,
        timestamp=avg_timestamp,
        latest_timestamp=avg_timestamp, 
        anomaly_arr=anom_arr_bin,
        cluster_lst=cluster_lst_bin,
    )

def update_FireModel(cluster, fire):
    """ 
    Parameters
    ----------
    cluster : np.array
        numpy array containing a list of anomalies FireModel
    fire : pages.models.FireModel
        firemodel we are updating
    """
    # Add new cluster to end of cluster list
    current_cluster_lst = binfield_to_obj(fire.cluster_lst)
    current_cluster_lst.append(cluster)    

    # Add current anomaly list to old anomaly list
    current_anomaly_arr = binfield_to_obj(fire.anomaly_arr)
    current_anomaly_arr = np.append(current_anomaly_arr, cluster.flatten(), 0)
    avg_timestamp = datetime.datetime.utcfromtimestamp(np.mean([dt.timestamp() for dt in cluster[:, 2]]))
    
    # Update fire model
    fire.cluster_lst = obj_to_binfield(current_cluster_lst)
    fire.anomaly_arr = obj_to_binfield(current_anomaly_arr)
    fire.latest_timestamp = avg_timestamp
    
    fire.save()

def update_picture():
    # TODO
    pass

def test_cluster():
    lons = np.random.randint(low=-120, high=-100, size=10)
    lats = np.random.randint(low=30, high=45, size=10)
    timestamp = datetime.datetime.utcnow()
    return np.array(list(zip(lons, lats, [timestamp for _ in lons])))
