# Rename this file to config.py and add the proper paths
import os

BASE_FOLDER = os.path.join('home', 'n', 'Documents', 'fnn-django')
NC_DATA_FOLDER_CALI = os.path.join(BASE_FOLDER, 'src', 'media', 'cali', 'data')
NC_DATA_FOLDER_BR = os.path.join(BASE_FOLDER, 'src', 'media', 'brazil', 'data')
API_KEY = os.path.join('home', 'n', 'Keys', 'fire-neural-network-3b8e8eff4400.json')
CALI_SHP_FOLDER = os.path.join(BASE_FOLDER, 'src', 'static', 'california', 'CA_State_TIGER2016.shp')
BR_SHP_FOLDER = os.path.join(BASE_FOLDER, 'src', 'static', 'brazil', 'Brazil_Boundary.shp')
MEDIA_FOLDER_CALI = os.path.join(BASE_FOLDER, 'src', 'media', 'cali')
MEDIA_FOLDER_BR = os.path.join(BASE_FOLDER, 'src', 'media', 'brazil')

GOOGLE_PROJECT_NAME = 'fire-neural-network'
ABI_SUBSCRIPTION_NAME_CALI = 'goes16-ABI-data-sub-filtered-cali'
GLM_SUBSCRIPTION_NAME_CALI = 'goes16-GLM-data-sub-filtered-cali'
ABI_SUBSCRIPTION_NAME_BR = 'goes16-ABI-data-sub-filtered-br'
GLM_SUBSCRIPTION_NAME_BR = 'goes16-GLM-data-sub-filtered-br'
SECRET_CONFIG_PATH = '/home/n/Keys/config_fnn.json'
