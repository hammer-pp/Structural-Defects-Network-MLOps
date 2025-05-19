import os
import cloudinary
import cloudinary.api
from airflow.sdk import Variable
from datetime import datetime
import logging
from dateutil import parser

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Configuring Cloudinary connection...")

# Fetch variables safely (Airflow 3.0+ allows default)
cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

# Check for missing configs and log error clearly
if not all([cloud_name, api_key, api_secret]):
    logger.error("Missing one or more Cloudinary credentials in Airflow Variables.")
else:
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )
    logger.info(f"Cloudinary configuration completed. {cloud_name}")

def check_new_images(**kwargs):
    # Load the last retraining time
    logger.info("Fetching the last retraining time from Airflow Variables...")
    last_retrain_str = Variable.get("last_retrain_time", default="2025-01-01T00:00:00")
    last_retrain_time = datetime.fromisoformat(last_retrain_str)
    logger.info(f"Last retraining time: {last_retrain_time}")

    # Fetch list of resources from Cloudinary's 'users/' folder
    logger.info("Fetching resources from Cloudinary 'users/' folder...")
    try:
        response = cloudinary.api.resources(type="upload", prefix="users/", max_results=500)
        resources = response['resources']
        logger.info(f"Successfully fetched {len(resources)} resources from Cloudinary.")
    except Exception as e:
        logger.error(f"Error fetching resources from Cloudinary: {e}")
        raise

    # Filter only images uploaded after last retrain
    logger.info("Filtering images uploaded after the last retraining time...")
    try:
        new_images = [
            img for img in resources 
            if parser.isoparse(img['created_at']) > last_retrain_time
        ]
        logger.info(f"Found {len(new_images)} new images in 'users/' after {last_retrain_time}.")
    except Exception as e:
        logger.error(f"Error filtering new images: {e}")
        raise

    # Decision based on the number of new images
    if len(new_images) >= 10:
        logger.info("Sufficient new images found. Proceeding to 'load_new_img'.")
        return 'load_new_img'
    else:
        logger.info("Not enough new images found. Proceeding to 'stop_no_data'.")
        return 'skip_training'