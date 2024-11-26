import os
import logging
import subprocess

class GCloudSync:

    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):
        # Ensure filepath ends with a slash for clarity
        if not filepath.endswith("/"):
            filepath += "/"

        # Construct the command
        command = f'gsutil cp "{filepath}{filename}" gs://{gcp_bucket_url}/'
        logging.info(f"Executing command: {command}")
        
        # Use subprocess for better error handling and logging
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Log the result
        if result.returncode == 0:
            logging.info(f"Successfully uploaded {filename} to gs://{gcp_bucket_url}/")
        else:
            logging.error(f"Failed to upload {filename} to gs://{gcp_bucket_url}/: {result.stderr}")
            raise Exception(f"gsutil cp failed: {result.stderr}")

    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):
        # Ensure the destination ends with a trailing slash
        if not destination.endswith("/"):
            destination += "/"

        command = f'gsutil cp gs://{gcp_bucket_url}/{filename} "{destination}"'
        logging.info(f"Executing command: {command}")
        
        # Use subprocess to run the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Successfully copied {filename} to {destination}")
        else:
            logging.error(f"Failed to copy {filename}: {result.stderr}")
            raise Exception(f"gsutil cp failed: {result.stderr}")