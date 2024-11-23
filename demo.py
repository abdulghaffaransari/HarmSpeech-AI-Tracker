from hate.logger import logging
from hate.exception import CustomException
import sys
from hate.configuration.gcloud_syncer import  GCloudSync

# logging.info("Welcome to our Project")


obj = GCloudSync()

obj.sync_folder_from_gcloud("hate-speech-nov-2024","dataset.zip","dataset.zip")