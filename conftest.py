import pytest
from hate.pipeline.train_pipeline import TrainPipeline

@pytest.fixture(scope="session", autouse=True)
def setup_pipeline():
    # Run the pipeline end-to-end once to ensure all artifacts are ready
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
