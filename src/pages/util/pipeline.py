# Standard Imports
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import datetime
import logging
import os
import pickle

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
        if band_id == 7:
            with open(os.path.join(config.NC_DATA_FOLDER, 'misc', 'classified_lst.pkl'), 'rb') as f:
                lst = pickle.load(f)
                if xds_path in lst:
                    return
                else:
                    ret = {'diff':xds_path, 'cloud':associate_path}
                    return ret
        elif band_id == 14:
            with open(os.path.join(config.NC_DATA_FOLDER, 'misc', 'classified_lst.pkl'), 'rb') as f:
                lst = pickle.load(f)
                if associate_path in lst:
                    return 
                else:
                    ret = {'diff':associate_path, 'cloud':xds_path}
                    return ret

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

def callback(message):
    """
    Parameters
    ----------
    message : str
        message recieved by subscriber

    Returns
    ----------

    """
    
    # Getting relavent bands 
    band_path = filter_band(message)


    if band_path[0] == None:
        message.ack()
        return 

    # Individual Band processes
    elif band_path[0] == 7:
        logging.info("Band 7 message recieved")
        path = band_7_process(band_path[1])


    elif band_path[0] == 14:
        logging.info("Band 14 message recieved")
        path = band_14_process(band_path[1])

    else:
        raise Exception("Unkonwn message recieved")

    if path == None:
        message.nack()
        return 
        
    bandpath_dct = get_associate_ABI(path)
    if bandpath_dct == None:
        message.ack()
        return
    else:
        predict.classify(bandpath_dct)


def pipeline():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.API_KEY
    logging.basicConfig(level=logging.INFO)

    project_id = "fire-neural-network"
    subscription_id = "goes16-ABI-data-sub-filtered"

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print("Listening for messages on {}..\n".format(subscription_path))

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()

# gcloud beta pubsub subscriptions create goes16-ABI-data-sub-filtered --project fire-neural-network --topic projects/gcp-public-data---goes-16/topics/gcp-public-data-goes-16 --message-filter='hasPrefix(attributes.objectId,"ABI-L1b-RadC/")' --enable-message-ordering
# gcloud pubsub subscriptions seek projects/fire-neural-network/subscriptions/goes16-ABI-data-sub-filtered --time=$(date +%Y-%m-%dT%H:%M:%S) 