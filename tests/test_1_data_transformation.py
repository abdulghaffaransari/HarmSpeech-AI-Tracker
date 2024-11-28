import os
import pytest
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts
from hate.constants import ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR

@pytest.fixture
def mock_data_ingestion_artifacts():
    # Dynamically generate paths based on the artifacts directory and constants
    artifacts_dir = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
    return DataIngestionArtifacts(
        imbalance_data_file_path=os.path.join(artifacts_dir, "imbalanced_data.csv"),
        raw_data_file_path=os.path.join(artifacts_dir, "raw_data.csv")
    )

@pytest.fixture
def data_transformation_config():
    # Dynamically generate the data transformation configuration
    return DataTransformationConfig()

def test_imbalance_data_cleaning(mock_data_ingestion_artifacts, data_transformation_config):
    data_transformation = DataTransformation(data_transformation_config, mock_data_ingestion_artifacts)
    try:
        imbalance_data = data_transformation.imbalance_data_cleaning()
        assert not imbalance_data.empty, "Imbalance data is empty after cleaning."
    except Exception as e:
        pytest.fail(f"Imbalance data cleaning failed: {e}")

def test_raw_data_cleaning(mock_data_ingestion_artifacts, data_transformation_config):
    data_transformation = DataTransformation(data_transformation_config, mock_data_ingestion_artifacts)
    try:
        raw_data = data_transformation.raw_data_cleaning()
        assert not raw_data.empty, "Raw data is empty after cleaning."
    except Exception as e:
        pytest.fail(f"Raw data cleaning failed: {e}")

def test_concat_dataframe(mock_data_ingestion_artifacts, data_transformation_config):
    data_transformation = DataTransformation(data_transformation_config, mock_data_ingestion_artifacts)
    try:
        df = data_transformation.concat_dataframe()
        assert not df.empty, "Concatenated dataframe is empty."
    except Exception as e:
        pytest.fail(f"Concatenating dataframes failed: {e}")

def test_data_transformation(mock_data_ingestion_artifacts, data_transformation_config):
    data_transformation = DataTransformation(data_transformation_config, mock_data_ingestion_artifacts)
    try:
        artifact = data_transformation.initiate_data_transformation()
        assert os.path.exists(artifact.transformed_data_path), "Transformed data file not created."
    except Exception as e:
        pytest.fail(f"Data transformation failed: {e}")
