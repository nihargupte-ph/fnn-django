import numpy as np
import os
import datetime
import time
import logging

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
    file_pair = [(parse_goes16_file(filename)[1], filename) for filename in os.listdir(folder)]
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