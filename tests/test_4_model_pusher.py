import os
import pytest
from hate.components.model_pusher import ModelPusher
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts


@pytest.fixture
def model_pusher_config():
    return ModelPusherConfig()


@pytest.fixture
def mock_model_trainer_artifacts():
    # Dynamically locate the latest artifacts directory
    artifacts_root = os.path.join("artifacts")
    if not os.path.exists(artifacts_root) or not os.listdir(artifacts_root):
        pytest.fail(f"No artifacts directory found at {artifacts_root}")

    # Get the latest directory based on timestamp
    latest_artifact_dir = sorted(os.listdir(artifacts_root))[-1]
    model_trainer_dir = os.path.join(artifacts_root, latest_artifact_dir, "ModelTrainerArtifacts")

    trained_model_path = os.path.join(model_trainer_dir, "model.h5")

    # Validate if the trained model exists
    if not os.path.exists(trained_model_path):
        pytest.fail(f"Trained model file not found at {trained_model_path}")

    return {"trained_model_path": trained_model_path}


def test_initiate_model_pusher(mock_model_trainer_artifacts, model_pusher_config):
    model_pusher = ModelPusher(model_pusher_config=model_pusher_config)
    try:
        artifacts = model_pusher.initiate_model_pusher()
        assert isinstance(artifacts, ModelPusherArtifacts), "Artifacts object not returned correctly."
    except Exception as e:
        pytest.fail(f"Model pushing failed: {e}")
