import os
import pytest
import pandas as pd
from hate.components.model_trainer import ModelTrainer
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import DataTransformationArtifacts
from hate.exception import CustomException

# Mock fixture for ModelTrainerConfig
@pytest.fixture
def model_trainer_config():
    return ModelTrainerConfig()

# Mock fixture for DataTransformationArtifacts
import os
import pytest
from hate.entity.artifact_entity import DataTransformationArtifacts

@pytest.fixture
def mock_data_transformation_artifacts():
    # Locate the latest timestamped artifacts directory dynamically
    artifacts_root = os.path.join("artifacts")
    if not os.path.exists(artifacts_root) or not os.listdir(artifacts_root):
        pytest.fail(f"No artifacts directory found at {artifacts_root}")
    
    # Get the latest directory based on timestamp
    latest_artifact_dir = sorted(os.listdir(artifacts_root))[-1]
    data_transformation_dir = os.path.join(artifacts_root, latest_artifact_dir, "DataTransformationArtifacts")
    final_csv_path = os.path.join(data_transformation_dir, "final.csv")
    
    # Validate if the file exists
    if not os.path.exists(final_csv_path):
        pytest.fail(f"Transformed data file 'final.csv' not found in {data_transformation_dir}")
    
    # Return the mock data transformation artifact
    return DataTransformationArtifacts(transformed_data_path=final_csv_path)


def test_splitting_data(model_trainer_config, mock_data_transformation_artifacts):
    model_trainer = ModelTrainer(
        data_transformation_artifacts=mock_data_transformation_artifacts,
        model_trainer_config=model_trainer_config
    )
    try:
        x_train, x_test, y_train, y_test = model_trainer.spliting_data(
            csv_path=mock_data_transformation_artifacts.transformed_data_path
        )
        assert len(x_train) > 0, "Training data is empty."
        assert len(x_test) > 0, "Test data is empty."
        assert len(y_train) > 0, "Training labels are empty."
        assert len(y_test) > 0, "Test labels are empty."
    except Exception as e:
        pytest.fail(f"Data splitting failed: {e}")

def test_tokenizing(model_trainer_config, mock_data_transformation_artifacts):
    model_trainer = ModelTrainer(
        data_transformation_artifacts=mock_data_transformation_artifacts,
        model_trainer_config=model_trainer_config
    )
    try:
        # Use splitting function to get data for tokenization
        x_train, _, _, _ = model_trainer.spliting_data(
            csv_path=mock_data_transformation_artifacts.transformed_data_path
        )
        sequences_matrix, tokenizer = model_trainer.tokenizing(x_train)
        assert sequences_matrix.shape[0] == len(x_train), "Tokenized sequences do not match training data size."
        assert tokenizer is not None, "Tokenizer is None."
    except Exception as e:
        pytest.fail(f"Tokenization failed: {e}")

def test_initiate_model_trainer(model_trainer_config, mock_data_transformation_artifacts):
    model_trainer = ModelTrainer(
        data_transformation_artifacts=mock_data_transformation_artifacts,
        model_trainer_config=model_trainer_config
    )
    try:
        artifacts = model_trainer.initiate_model_trainer()
        assert os.path.exists(artifacts.trained_model_path), "Trained model file not found."
        assert os.path.exists(artifacts.x_test_path), "Test data file not found."
        assert os.path.exists(artifacts.y_test_path), "Test labels file not found."
    except Exception as e:
        pytest.fail(f"Model training failed: {e}")
