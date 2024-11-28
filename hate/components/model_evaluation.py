import os
import sys
import keras
import pickle
import pandas as pd
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info("[GCLOUD] Fetching the best model from GCloud storage")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(
                self.model_evaluation_config.BUCKET_NAME,
                self.model_evaluation_config.MODEL_NAME,
                self.model_evaluation_config.BEST_MODEL_DIR_PATH
            )

            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info(f"[GCLOUD] Best model fetched: {best_model_path}")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_data(self):
        try:
            logging.info("[PREPROCESS] Loading x_test and y_test data")
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            logging.info("[PREPROCESS] Validating column names in x_test")
            x_test.columns = x_test.columns.str.strip()
            if 'tweet' not in x_test.columns:
                raise ValueError("[PREPROCESS] Column 'tweet' not found in x_test. Verify preprocessing.")

            x_test['tweet'] = x_test['tweet'].astype(str).str.strip()

            logging.info("[PREPROCESS] Validating data cardinality between x_test and y_test")
            if len(x_test) != len(y_test):
                logging.warning("[PREPROCESS] Data mismatch detected. Aligning x_test and y_test based on indices.")
                min_size = min(len(x_test), len(y_test))
                x_test = x_test.iloc[:min_size]
                y_test = y_test.iloc[:min_size]

            logging.info("[PREPROCESS] Loading tokenizer for text processing")
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            logging.info("[PREPROCESS] Tokenizing and padding x_test")
            test_sequences = tokenizer.texts_to_sequences(x_test['tweet'])
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            return test_sequences_matrix, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self, x_test, y_test, model_path: str, model_type: str):
        try:
            logging.info(f"[{model_type}] Loading model from {model_path}")
            model = keras.models.load_model(model_path)

            logging.info(f"[{model_type}] Evaluating the model")
            accuracy = model.evaluate(x_test, y_test, verbose=1)
            logging.info(f"[{model_type}] Accuracy on test data: {accuracy}")

            logging.info(f"[{model_type}] Generating confusion matrix")
            predictions = model.predict(x_test)
            predictions = [0 if pred[0] < 0.5 else 1 for pred in predictions]
            confusion = confusion_matrix(y_test, predictions)
            logging.info(f"[{model_type}] Confusion matrix:\n{confusion}")

            return accuracy, confusion
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("[EVALUATION] Starting model evaluation process")

            logging.info("[EVALUATION] Preprocessing data for evaluation")
            x_test, y_test = self.preprocess_data()

            trained_model_path = self.model_trainer_artifacts.trained_model_path
            logging.info("[EVALUATION] Evaluating the trained model")
            trained_model_accuracy, trained_confusion = self.evaluate(
                x_test, y_test, model_path=trained_model_path, model_type="TRAINED MODEL"
            )

            logging.info("[EVALUATION] Fetching the best model from GCloud storage")
            best_model_path = self.get_best_model_from_gcloud()

            is_model_accepted = False
            if not os.path.isfile(best_model_path):
                logging.info("[EVALUATION] No best model found in GCloud. Accepting the trained model.")
                is_model_accepted = True
            else:
                logging.info("[EVALUATION] Evaluating the best model fetched from GCloud")
                best_model_accuracy, best_confusion = self.evaluate(
                    x_test, y_test, model_path=best_model_path, model_type="BEST MODEL FROM GCLOUD"
                )

                logging.info("[EVALUATION] Comparing trained model with the best model")
                if trained_model_accuracy[1] >= best_model_accuracy[1]:  # Compare based on accuracy
                    is_model_accepted = True
                    logging.info("[EVALUATION] Trained model is better or equal to the best model. Accepting it.")
                else:
                    logging.info("[EVALUATION] Trained model is not better than the best model.")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("[EVALUATION] Model evaluation process completed successfully")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
