import os
import sys
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of DataIngestion class")

            # Create artifacts directory
            logging.info(f"Creating directory: {self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            # Sync data from GCloud
            logging.info(f"Syncing {self.data_ingestion_config.ZIP_FILE_NAME} from bucket {self.data_ingestion_config.BUCKET_NAME} "
                         f"to {self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")
            self.gcloud.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_NAME,
                                                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            
            # Verify if file exists
            zip_file_path = self.data_ingestion_config.ZIP_FILE_PATH
            if os.path.exists(zip_file_path):
                logging.info(f"File successfully synced to {zip_file_path}")
            else:
                logging.error(f"File not found at {zip_file_path} after syncing")
                raise FileNotFoundError(f"File not found: {zip_file_path}")

            logging.info("Exited the get_data_from_gcloud method of DataIngestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of DataIngestion class")
        zip_file_path = self.data_ingestion_config.ZIP_FILE_PATH
        zip_file_dir = self.data_ingestion_config.ZIP_FILE_DIR

        try:
            # Verify if ZIP file exists
            if not os.path.exists(zip_file_path):
                logging.error(f"ZIP file does not exist at {zip_file_path}")
                raise FileNotFoundError(f"ZIP file not found at {zip_file_path}")
            
            logging.info(f"Unzipping {zip_file_path} to {zip_file_dir}")
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_file_dir)
            
            # Verify extracted files
            extracted_files = os.listdir(zip_file_dir)
            if extracted_files:
                logging.info(f"Extracted files: {extracted_files}")
            else:
                logging.error(f"No files extracted to {zip_file_dir}")
                raise FileNotFoundError(f"No files found in extracted directory {zip_file_dir}")

            logging.info("Exited the unzip_and_clean method of DataIngestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")

        try:
            # Fetch data from GCloud
            self.get_data_from_gcloud()
            logging.info("Successfully fetched data from GCloud")

            # Unzip and clean data
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Successfully unzipped and cleaned data")

            # Create DataIngestionArtifacts
            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            logging.info(f"Data ingestion artifacts created: {data_ingestion_artifacts}")
            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
