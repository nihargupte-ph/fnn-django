# Standard imports
import os
import xarray
import logging
import sys

# Custom imports
from pages.util import config 
from pages.util import misc_functions
from pages.util.predict import clip_xds_cali, normalize_xds


def parse_blobpath(blob_path):
  blob_path = blob_path.replace('%2F', '/')
  blob_path = blob_path[3:]  # /b/
  slash_loc = blob_path.index('/')
  bucket_name = blob_path[:slash_loc]
  blob_name = blob_path[(slash_loc + 3):]  # /o/
  return bucket_name, blob_name

def path_to_blob(blob_path, gcs_client):
  """ Given blob_path returns blob object """

  bucket, object_id = parse_blobpath(blob_path)
  bucket = gcs_client.get_bucket(bucket)
  blob = bucket.blob(object_id)

  return blob, object_id

def copy_fromgcs(bucket, objectId, destdir):
   import os.path
   import google.cloud.storage as gcs
   bucket = gcs.Client().get_bucket(bucket)
   blob = bucket.blob(objectId)
   basename = os.path.basename(objectId)
   dest = os.path.join(destdir, basename)
   blob.download_to_filename(dest)
   return dest

def download_preproc_ABI(objectId, area, band_id=None):
    """
    Parameters
    ----------
    objectId : str
        objectId of object in gcs bucket
    area : dict
        contains everything about the area (folders, shape file, model)
    band_id : int
        id of band (7 or 14 it could be None)

    Returns
    ----------
    filepath : str
        filepath
    """
    if band_id == None:
        if 'C14_G16' in objectId:
            band_id = 14
        elif 'C07_G16' in objectId:
            band_id = 7

    # Download 
    try: 
        filepath = copy_fromgcs("gcp-public-data-goes-16", objectId, os.path.join(area['data_folder'], area['bucket'], 'actual', f"band_{band_id}"))
        logging.info(f"Successfully downloaded {objectId} from goes-16 {area['bucket']} bucket")
    except:
        logging.critical(f"File with objectId:{objectId} was not able to be Downloaded\n" + str(misc_functions.error_handling()))
        sys.exit(1)

    # Normalize
    try:
        xds = xarray.open_dataset(filepath)
        xds_copy = xds.copy()
        xds_copy = normalize_xds(xds_copy)
        logging.info(f"Normalized {objectId}")
    except:
        logging.critical(f"File with objectId:{objectId} was not able to be Normalized\n" + str(misc_functions.error_handling()))
        raise Exception

    # Clip
    try:
        clipped_xds = clip_xds_cali(xds_copy, area['shape'])
        # We need to remove it and not just overwrite as overwriting netcdf4 files requires elevated priveleges
        xds.close()
        os.remove(filepath)
        clipped_xds.to_netcdf(path=filepath, mode='w')
        logging.info(f"Clipped {objectId}")
    except: 
        logging.critical(f"File with objectId:{objectId} was not able to be Clipped\n" + str(misc_functions.error_handling()))
        raise Exception

    clipped_xds.close()
    xds_copy.close()

    return filepath

def download_GLM_goes16_data(objectId, area):
    """
    Parameters
    ----------
    objectId : str
        objectId of object in gcs bucket
    area : dict
        contains everything about the area (folders, shape file, model)
        
    Returns
    ----------
    filepath : str
        filepath
    """
    try:
        filepath = copy_fromgcs("gcp-public-data-goes-16", objectId, os.path.join(area['data_folder'], "GLM"))
    except:
        logging.warn("unable to download from GLM bucket")
        return

    return filepath