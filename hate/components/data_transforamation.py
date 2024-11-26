import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from hate.logger import logging 
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def imbalance_data_cleaning(self):
        try:
            logging.info("Cleaning imbalance data...")
            imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            imbalance_data.drop(
                self.data_transformation_config.ID,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )
            logging.info("Imbalance data cleaning completed.")
            return imbalance_data
        except Exception as e:
            raise CustomException(e, sys) from e

    def raw_data_cleaning(self):
        try:
            logging.info("Cleaning raw data...")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            raw_data.drop(
                self.data_transformation_config.DROP_COLUMNS,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )
            raw_data[self.data_transformation_config.CLASS].replace({0: 1, 2: 0}, inplace=True)
            raw_data.rename(
                columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL},
                inplace=True
            )
            logging.info("Raw data cleaning completed.")
            return raw_data
        except Exception as e:
            raise CustomException(e, sys) from e

    def concat_dataframe(self):
        try:
            logging.info("Concatenating dataframes...")
            frame = [self.raw_data_cleaning(), self.imbalance_data_cleaning()]
            df = pd.concat(frame)
            logging.info("Dataframe concatenation completed.")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def concat_data_cleaning(self, words):
        try:
            # Internal function for actual cleaning logic (no logs here)
            def clean_text(text):
                stemmer = nltk.SnowballStemmer("english")
                stopword = set(stopwords.words('english'))
                text = str(text).lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w*\d\w*', '', text)
                text = [word for word in text.split(' ') if word not in stopword]
                text = " ".join(text)
                text = [stemmer.stem(word) for word in text.split(' ')]
                return " ".join(text)

            return clean_text(words)
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Starting data transformation process...")
            df = self.concat_dataframe()

            # Log before and after applying the cleaning function
            logging.info("Applying text cleaning to the entire dataframe...")
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(
                self.concat_data_cleaning
            )
            logging.info("Text cleaning completed for the entire dataframe.")

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            logging.info("Data transformation process completed. Saving transformed data.")
            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("Returning DataTransformationArtifacts.")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

