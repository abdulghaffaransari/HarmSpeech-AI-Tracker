import os
import pytest
from hate.components.data_ingestion import DataIngestion
from hate.entity.config_entity import DataIngestionConfig

@pytest.fixture
def data_ingestion_config():
    return DataIngestionConfig()

def test_get_data_from_gcloud(data_ingestion_config):
    data_ingestion = DataIngestion(data_ingestion_config)
    try:
        data_ingestion.get_data_from_gcloud()
        assert os.path.exists(data_ingestion_config.ZIP_FILE_PATH), "Zip file not downloaded."
    except Exception as e:
        pytest.fail(f"Data ingestion from GCloud failed: {e}")

def test_unzip_and_clean(data_ingestion_config):
    data_ingestion = DataIngestion(data_ingestion_config)
    try:
        imbalance_data, raw_data = data_ingestion.unzip_and_clean()
        assert os.path.exists(imbalance_data), "Imbalance data file not found."
        assert os.path.exists(raw_data), "Raw data file not found."
    except Exception as e:
        pytest.fail(f"Data unzip and cleaning failed: {e}")
