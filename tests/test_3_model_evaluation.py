import os
import pytest
from hate.components.model_evaluation import ModelEvaluation
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts


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
    
    return DataTransformationArtifacts(transformed_data_path=final_csv_path)


@pytest.fixture
def mock_model_trainer_artifacts():
    # Locate the latest timestamped artifacts directory dynamically
    artifacts_root = os.path.join("artifacts")
    if not os.path.exists(artifacts_root) or not os.listdir(artifacts_root):
        pytest.fail(f"No artifacts directory found at {artifacts_root}")
    
    # Get the latest directory based on timestamp
    latest_artifact_dir = sorted(os.listdir(artifacts_root))[-1]
    model_trainer_dir = os.path.join(artifacts_root, latest_artifact_dir, "ModelTrainerArtifacts")
    
    trained_model_path = os.path.join(model_trainer_dir, "model.h5")
    x_test_path = os.path.join(model_trainer_dir, "x_test.csv")
    y_test_path = os.path.join(model_trainer_dir, "y_test.csv")
    
    # Validate if files exist
    for file_path in [trained_model_path, x_test_path, y_test_path]:
        if not os.path.exists(file_path):
            pytest.fail(f"File not found: {file_path}")
    
    return ModelTrainerArtifacts(
        trained_model_path=trained_model_path,
        x_test_path=x_test_path,
        y_test_path=y_test_path
    )


@pytest.fixture
def model_evaluation_config():
    return ModelEvaluationConfig()


def test_get_best_model_from_gcloud(model_evaluation_config):
    model_evaluation = ModelEvaluation(
        model_evaluation_config=model_evaluation_config,
        model_trainer_artifacts=None,  # Not required for this test
        data_transformation_artifacts=None  # Not required for this test
    )
    try:
        best_model_path = model_evaluation.get_best_model_from_gcloud()
        assert os.path.exists(best_model_path), f"Best model file not found at {best_model_path}"
    except Exception as e:
        pytest.fail(f"Failed to fetch best model from GCloud: {e}")


def test_preprocess_data(mock_model_trainer_artifacts, mock_data_transformation_artifacts, model_evaluation_config):
    model_evaluation = ModelEvaluation(
        model_evaluation_config=model_evaluation_config,
        model_trainer_artifacts=mock_model_trainer_artifacts,
        data_transformation_artifacts=mock_data_transformation_artifacts
    )
    try:
        x_test, y_test = model_evaluation.preprocess_data()
        assert x_test is not None and len(x_test) > 0, "x_test is empty or None"
        assert y_test is not None and len(y_test) > 0, "y_test is empty or None"
    except Exception as e:
        pytest.fail(f"Data preprocessing failed: {e}")


def test_evaluate(mock_model_trainer_artifacts, mock_data_transformation_artifacts, model_evaluation_config):
    model_evaluation = ModelEvaluation(
        model_evaluation_config=model_evaluation_config,
        model_trainer_artifacts=mock_model_trainer_artifacts,
        data_transformation_artifacts=mock_data_transformation_artifacts
    )
    try:
        x_test, y_test = model_evaluation.preprocess_data()
        trained_model_path = mock_model_trainer_artifacts.trained_model_path
        accuracy, confusion = model_evaluation.evaluate(
            x_test=x_test,
            y_test=y_test,
            model_path=trained_model_path,
            model_type="TRAINED MODEL"
        )
        assert accuracy is not None, "Evaluation accuracy is None"
        assert confusion is not None, "Confusion matrix is None"
    except Exception as e:
        pytest.fail(f"Model evaluation failed: {e}")


def test_initiate_model_evaluation(mock_model_trainer_artifacts, mock_data_transformation_artifacts, model_evaluation_config):
    model_evaluation = ModelEvaluation(
        model_evaluation_config=model_evaluation_config,
        model_trainer_artifacts=mock_model_trainer_artifacts,
        data_transformation_artifacts=mock_data_transformation_artifacts
    )
    try:
        artifacts = model_evaluation.initiate_model_evaluation()
        assert artifacts is not None, "Model evaluation artifacts are None"
        assert isinstance(artifacts.is_model_accepted, bool), "is_model_accepted is not a boolean"
    except Exception as e:
        pytest.fail(f"Model evaluation initiation failed: {e}")
