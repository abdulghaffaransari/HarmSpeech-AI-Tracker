import os
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data from path: %s", csv_path)
            df = pd.read_csv(csv_path, index_col=False)
            
            logging.info("Splitting the data into X (tweets) and Y (labels)")
            x = df[TWEET].fillna('').astype(str)  # Ensure all text data is valid
            y = df[LABEL]

            logging.info("Performing train-test split")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            
            logging.info(f"Train data size: {len(x_train)}, Test data size: {len(x_test)}")
            logging.info("Exited the spliting_data function")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, x_train):
        try:
            logging.info("Entered the tokenizing function")
            logging.info("Applying tokenization on the training data")
            
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            
            logging.info("Converting texts to sequences")
            sequences = tokenizer.texts_to_sequences(x_train)
            
            logging.info("Padding the sequences to uniform length")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            
            logging.info("Tokenization and padding completed")
            logging.info("Exited the tokenizing function")
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logging.info("Splitting data into training and testing sets")
            x_train, x_test, y_train, y_test = self.spliting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )

            logging.info("Initializing model architecture")
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            logging.info("Preparing training data for the model")
            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Starting model training")
            model.fit(
                sequences_matrix,
                y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
            )
            logging.info("Model training completed")

            logging.info("Saving tokenizer")
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            logging.info("Saving the trained model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info("Saving test and training data")
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH, index=False)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH, index=False)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH, index=False)

            logging.info("Creating model trainer artifacts")
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )
            
            logging.info("Model trainer artifacts created successfully")
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
