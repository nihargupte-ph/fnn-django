# Standard Imports
import datetime
import logging
import os
import pickle

# Third Party imports
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import tensorflow as tf

# Custom Imports
from pages.util import config
from pages.util import misc_functions
from pages.util.nn import nn7
from pages.util import predict
from pages.util import download

def get_associate_ABI(xds_path):
    """
    Parameters
    ----------
    xds_path : str
        File path which we want the associated ABI filepath of. For example, if xds_path is 
        for a band 7 file we would return a band 14 file with the same timesatmp. Note that 
        if the xds has already been classified according to media/classified_files.txt then
        this function will return None since we dont' want to predict twice

    Returns
    ----------
    ret : dict
        dict of filepaths one entry is for diff and the other for cloud. Keys are int values are str
        Will return None if the associated filepath is not found or if we have already used 
        the band 7 file to classify fires
    """
    orig_key = misc_functions.key_from_filestring(xds_path)

    if 'C07_G16' in xds_path:
        band_id = 7
        associate_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud')
        associate_file_lst = os.listdir(associate_folder)
    elif 'C14_G16' in xds_path:
        band_id = 14
        associate_folder = os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff')
        associate_file_lst = os.listdir(associate_folder)
    else:
        logging.critical("band id not found in path")

    associate_key_lst = [misc_functions.key_from_filestring(file_string) for file_string in associate_file_lst]
    if orig_key in associate_key_lst:
        associate_idx = associate_key_lst.index(orig_key)
        associate_file = associate_file_lst[associate_idx]
        associate_path = os.path.join(associate_folder, associate_file)

        # If we have already used the file to classify then we don't want to reuse
        # We determine if we already have the file by checking the classified_lst 
        with open(os.path.join(config.MEDIA_FOLDER, 'misc', 'classified_lst.pkl'), 'rb') as f:
            classified_lst = pickle.load(f)
        if band_id == 7:
            if os.path.basename(xds_path) in classified_lst:
                return 
            else:
                logging.info("Got associate path")
                ret = {'diff':xds_path, 'cloud':associate_path}
        elif band_id == 14:
            if os.path.basename(associate_path) in classified_lst:
                return 
            else:
                logging.info("Got associate path")
                ret = {'diff':associate_path, 'cloud':xds_path}

        return ret

    # We don't have the associate path yet
    return 

def band_7_process(objectId):
    """
    Parameters
    ----------
    objectId : str
        ObjectId in google bucket 

    Returns
    ----------
    filepath : str
       path to predicted xarray
    """
    xds_path = download.download_preproc_ABI(objectId)

    if xds_path == None:
        return 

    filepath = predict.predict_xarray(xds_path)

    return filepath

def band_14_process(objectId):
    """
    Parameters
    ----------
    objectId : str
        ObjectId in google bucket 

    Returns
    ----------
    filepath : str
       path to cloud array
    """
    xds_path = download.download_preproc_ABI(objectId)

    if xds_path == None:
        return 

    filepath = predict.predict_cloud(xds_path)

    return filepath

def filter_band(message):
    """
    Parameters
    ----------
    message : str
        message recieved by subscriber

    Returns
    ----------
    ret : (band_id or None, objectId)
        Returns None if it's a band we don't care about and returns the band 
        along with objectId if we care about it.
    """
    objectId = message.attributes.get('objectId')
    if 'C14_G16' in objectId and 'ABI-L1b-RadC' in objectId:
        ret = (14, objectId)
        return ret 
    elif 'C07_G16' in objectId and 'ABI-L1b-RadC' in objectId:
        ret = (7, objectId)
        return ret
    else:
        ret = (None, objectId)
        return ret

def clear_folders():
    """ 
    Clears all folders before starting pipeline
    """
    folder_lst = [
    os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_7'), 
    os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', 'band_14'), 
    os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'diff'), 
    os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'pred'), 
    os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'pred', 'cloud'),
    ]

    for folder in folder_lst:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

def callback(message):
    """
    Parameters
    ----------
    message : str
        message recieved by subscriber

    Returns
    ----------

    """
    
    # NOTE TEMP
    with open("/home/n/Documents/Research/fnn-django/src/media/misc/classified_lst.pkl", "wb") as f:
        pickle.dump([], f)

    # Getting relavent bands 
    # band_path = filter_band(message)
    band_path = (7, 'ABI-L1b-RadC/2020/240/00/OR_ABI-L1b-RadC-M6C07_G16_s20202400056172_e20202400058557_c20202400058597.nc')

    if band_path[0] == None:
        message.ack()
        return 

    # Individual Band processes
    elif band_path[0] == 7:
        logging.info("Band 7 message recieved")
        path = band_7_process(band_path[1])
        # message.ack()

    elif band_path[0] == 14:
        logging.info("Band 14 message recieved")
        path = band_14_process(band_path[1])
        # message.ack()

    else:
        raise Exception("Unkonwn message recieved")

    if path == None:
        message.nack()
        return 
    
    bandpath_dct = get_associate_ABI(path)
    # bandpath_dct = {
    #     'diff': '/home/n/Documents/Research/fnn-django/src/media/data/ABI_RadC/pred/diff/OR_ABI-L1b-RadC-M6C07_G16_s20202400056172_e20202400058557_c20202400058597.nc',
    #     'cloud': '/home/n/Documents/Research/fnn-django/src/media/data/ABI_RadC/pred/cloud/OR_ABI-L1b-RadC-M6C14_G16_s20202400056172_e20202400058545_c20202400059079.nc'
    # }
    if bandpath_dct == None:
        return
    else:
        predict.classify(bandpath_dct)
        return

def pipeline():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.API_KEY
    logging.basicConfig(level=logging.INFO)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    callback('hi')

    # project_id = "fire-neural-network"
    # subscription_id = "goes16-ABI-data-sub-filtered"

    # subscriber = pubsub_v1.SubscriberClient()
    # subscription_path = subscriber.subscription_path(project_id, subscription_id)
    # streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    # try: 
    #     clear_folders()
    #     logging.info("Successfully cleared folders")
    # except:
    #     logging.critical("Unable to clear folders")
    # logging.info("Listening for messages on {}..\n".format(subscription_path))

    # # Wrap subscriber in a 'with' block to automatically call close() when done.
    # with subscriber:
    #     try:
    #         # When `timeout` is not set, result() will block indefinitely,
    #         # unless an exception is encountered first.
    #         streaming_pull_future.result()
    #     except TimeoutError:
    #         streaming_pull_future.cancel()

# gcloud beta pubsub subscriptions create goes16-ABI-data-sub-filtered --project fire-neural-network --topic projects/gcp-public-data---goes-16/topics/gcp-public-data-goes-16 --message-filter='hasPrefix(attributes.objectId,"ABI-L1b-RadC/")' --enable-message-ordering
# gcloud pubsub subscriptions seek projects/fire-neural-network/subscriptions/goes16-ABI-data-sub-filtered --time=$(date +%Y-%m-%dT%H:%M:%S) 