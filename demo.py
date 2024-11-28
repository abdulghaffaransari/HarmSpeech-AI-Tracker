# from hate.logger import logging
# from hate.exception import CustomException
# import sys
# from hate.configuration.gcloud_syncer import  GCloudSync

# # logging.info("Welcome to our Project")


# obj = GCloudSync()

# obj.sync_folder_from_gcloud("hate-speech-nov-2024","dataset.zip","dataset.zip")



from hate.pipeline.prediction_pipeline import PredictionPipeline

# Example text for prediction
input_text = "This is an example tweet containing abusive language."

# Initialize PredictionPipeline and run it
try:
    pipeline = PredictionPipeline()
    result = pipeline.run_pipeline(input_text)
    print(f"Prediction Result: {result}")
except Exception as e:
    print(f"Error occurred: {str(e)}")
