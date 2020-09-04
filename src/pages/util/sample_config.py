# Rename this file to config.py and add the proper paths
import os

BASE_FOLDER = os.path.join('home', 'n', 'Documents', 'fnn-django')
NC_DATA_FOLDER = os.path.join(BASE_FOLDER, 'src', 'media', 'data')
API_KEY = os.path.join('home', 'n', 'Keys', 'fire-neural-network-3b8e8eff4400.json')
CALI_SHP_FOLDER = os.path.join(BASE_FOLDER, 'src', 'static', 'california')
MEDIA_FOLDER = os.path.join(BASE_FOLDER, 'src', 'media')

GOOGLE_PROJECT_NAME = 'fire-neural-network'
ABI_SUBSCRIPTION_NAME = 'goes16-ABI-data-sub-filtered'
GLM_SUBSCRIPTION_NAME = 'goes16-GLM-data-sub-filtered'