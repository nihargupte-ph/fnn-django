# Standard imports
import os
import xarray
import logging

# Custom imports
from detect_fires import config
from detect_fires.predict import clip_xds_cali, normalize_xds


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

def download_preproc_ABI(objectId):
    """
    Parameters
    ----------
    objectId : str
        objectId of object in gcs bucket

    Returns
    ----------
    filepath : str
        filepath
    """
    if 'C14_G16' in objectId:
        band_id = 14
    elif 'C07_G16' in objectId:
        band_id = 7

    # Download 
    try: 
        filepath = copy_fromgcs("gcp-public-data-goes-16", objectId, os.path.join(config.NC_DATA_FOLDER, 'ABI_RadC', 'actual', f"band_{band_id}"))
        logging.info(f"Successfully downloaded {objectId} from goes-16 ABI_RadC bucket")
    except:
        logging.warn(f"file with objectId:{objectId} was not able to be downloaded")
        return None

    # Normalize
    try:
        xds = xarray.open_dataset(filepath)
        xds_copy = xds.copy()
        xds_copy = normalize_xds(xds_copy)
        logging.info(f"Normalized {objectId}")
    except:
        logging.warn(f"file with objectId:{objectId} was not able to be normalized. Deleting and continuing...")
        os.remove(filepath)
        return None

    # Clip
    try:
        clipped_xds = clip_xds_cali(xds_copy)
        # We need to remove it and not just overwrite as overwriting netcdf4 files requires elevated priveleges
        os.remove(filepath)
        clipped_xds.to_netcdf(path=filepath, mode='w')
        logging.info(f"Clipped {objectId}")
    except: 
        logging.warn(f"file with objectId:{objectId} was not able to be clipped. Deleting and continuing...")
        os.remove(filepath)
        return None

    clipped_xds.close()
    xds.close()
    xds_copy.close()

    return filepath


def download_GLM_goes16_data(objectId):
    """
    Parameters
    ----------
    objectId : str
        objectId of object in gcs bucket

    Returns
    ----------
    filepath : str
        filepath
    """
    try:
        filepath = copy_fromgcs("gcp-public-data-goes-16", objectId, os.path.join(config.NC_DATA_FOLDER, "GLM"))
        logging.info("Successfully downloaded from GLM bucket")
    except:
        logging.warn("unable to download from GLM bucket")
        return

    return filepath

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.API_KEY
# donwload_goes16_ABIRadC_data("ABI-L1b-RadC/2020/222/03/OR_ABI-L1b-RadC-M6C14_G16_s20202220351190_e20202220353563_c20202220354014.nc")
