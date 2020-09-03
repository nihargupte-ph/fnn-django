# Standard Imports
import datetime
import logging
import os
import pickle
import threading

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
    os.path.join(config.NC_DATA_FOLDER, 'GLM'),
    os.path.join(config.MEDIA_FOLDER, 'tmp')
    ]

    for folder in folder_lst:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

def callback_ABI(message):
    """
    Parameters
    ----------
    message : str
        message recieved by subscriber. Handles messages from ABI subscription

    Returns
    ----------
    None 
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
        message.ack()

    elif band_path[0] == 14:
        logging.info("Band 14 message recieved")
        path = band_14_process(band_path[1])
        message.ack()

    else:
        raise Exception("Unkonwn message recieved")

    if path == None:
        message.nack()
        return 
    
    
    # with open("/home/n/Documents/Research/fnn-django/src/media/misc/classified_lst.pkl", "wb") as f:
    #     pickle.dump([], f)
    # path = message
    bandpath_dct = get_associate_ABI(path)

    if bandpath_dct == None:
        return
    else:
        predict.classify(bandpath_dct)
        return

def callback_GLM(message):
    """
    Parameters
    ----------
    message : str
        message recieved by subscriber. Handles messages from GLM subscription

    Returns
    ----------
    None
    """

    objectId = message.attributes.get('objectId')
    download.download_GLM_goes16_data(objectId)

    # Dealing with cleanup, we don't want any more than the specified number of files in the GLM folder. 
    file_lst = os.listdir(os.path.join(config.NC_DATA_FOLDER, 'GLM'))
    if len(file_lst) > 100: # this number times 1 minute of data. THIS is how long our search window is
        oldest_file = min(file_lst, key=lambda x: misc_functions.key_from_filestring(x))
        os.remove(os.path.join(config.NC_DATA_FOLDER, 'GLM', oldest_file))


def pipeline():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.API_KEY
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # diff_folder, cloud_folder = '/home/n/Documents/Research/fnn-django/src/media/data/ABI_RadC/pred/diff', '/home/n/Documents/Research/fnn-django/src/media/data/ABI_RadC/pred/cloud'
    # diff_lst, cloud_lst = os.listdir(diff_folder), os.listdir(cloud_folder)
    # diff_lst = sorted(diff_lst, key=misc_functions.key_from_filestring)
    # for diff_file in diff_lst[:2]:
    #     import xarray
    #     xds = xarray.open_dataset(os.path.join(diff_folder, diff_file))
    #     print(f'tstart  {xds.t.values}')
    #     callback_ABI(os.path.join(diff_folder, diff_file))


    project_id = "fire-neural-network"
    subscription_id1 = "goes16-ABI-data-sub-filtered"
    subscription_id2 = "goes16-GLM-data-sub-filtered"

    subscriber1 = pubsub_v1.SubscriberClient()
    subscriber2 = pubsub_v1.SubscriberClient()
    subscription_path1 = subscriber1.subscription_path(project_id, subscription_id1)
    streaming_pull_future1 = subscriber1.subscribe(subscription_path1, callback=callback_ABI)
    subscription_path2 = subscriber2.subscription_path(project_id, subscription_id2)
    streaming_pull_future2 = subscriber2.subscribe(subscription_path2, callback=callback_GLM)
    try: 
        clear_folders()
        logging.info("Successfully cleared folders")
    except:
        logging.critical("Unable to clear folders")
    logging.info(f"Listening for messages on {subscription_path1} and {subscription_path2}..\n")

    subscriber_shutdown = threading.Event()
    streaming_pull_future1.add_done_callback(lambda result: subscriber_shutdown.set())
    streaming_pull_future2.add_done_callback(lambda result: subscriber_shutdown.set())


    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber1, subscriber2:
        subscriber_shutdown.wait()
        streaming_pull_future1.cancel()
        streaming_pull_future2.cancel()

# gcloud beta pubsub subscriptions create goes16-ABI-data-sub-filtered --project fire-neural-network --topic projects/gcp-public-data---goes-16/topics/gcp-public-data-goes-16 --message-filter='hasPrefix(attributes.objectId,"ABI-L1b-RadC/")' --enable-message-ordering
# gcloud beta pubsub subscriptions create goes16-GLM-data-sub-filtered --project fire-neural-network --topic projects/gcp-public-data---goes-16/topics/gcp-public-data-goes-16 --message-filter='hasPrefix(attributes.objectId,"GLM-L2-LCFA/")' --enable-message-ordering
# gcloud pubsub subscriptions seek projects/fire-neural-network/subscriptions/goes16-ABI-data-sub-filtered --time=$(date +%Y-%m-%dT%H:%M:%S) 
# gcloud pubsub subscriptions seek projects/fire-neural-network/subscriptions/goes16-GLM-data-sub-filtered --time=$(date +%Y-%m-%dT%H:%M:%S) 