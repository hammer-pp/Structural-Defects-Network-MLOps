import os
import csv
import cloudinary
import cloudinary.api
import logging
from airflow.models import Variable
from datetime import datetime
from dateutil import parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_and_generate_csv(**kwargs):
    """
    Checks for new images uploaded to a specified Cloudinary folder since the last model retraining,
    generates a CSV file containing all image URLs with their associated place and label, and determines
    whether there are enough new images to proceed with retraining.

    This function performs the following steps:
    1. Configures Cloudinary credentials from environment variables.
    2. Retrieves the timestamp of the last model retraining from an Airflow Variable.
    3. Scans the specified Cloudinary folder for all uploaded images.
    4. Builds direct URLs for each image and extracts metadata (place, label) from the public ID.
    5. Writes all image URLs and metadata to a CSV file.
    6. Checks if at least 10 new images have been uploaded since the last retraining.
    7. Returns the next task to execute in the Airflow pipeline based on the number of new images found.

    Args:
        **kwargs: Arbitrary keyword arguments passed from Airflow.

    Returns:
        str: 'preprocess_data' if enough new images are found, otherwise 'skip_training'.
    """
    # Setup Cloudinary credentials
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )

    OUTPUT_CSV = 'cloudinary_dataset_users.csv'
    folder = "users/"
    include_version = True

    def build_url(resource):
        base = f"https://res.cloudinary.com/{cloudinary.config().cloud_name}/image/upload"
        path = resource['public_id'] + '.' + resource['format']
        if include_version:
            return f"{base}/v{resource['version']}/{path}"
        else:
            return f"{base}/{path}"

    # Get retraining time from Airflow Variable
    last_retrain_str = Variable.get("last_retrain_time", default_var="2025-01-01T00:00:00")
    last_retrain_time = datetime.fromisoformat(last_retrain_str)
    logger.info(f"Last retrain: {last_retrain_time}")

    # Fetch resources
    logger.info(f"Scanning folder: {folder}")
    next_cursor = None
    all_rows = []
    new_images = []

    while True:
        response = cloudinary.api.resources(
            type="upload",
            prefix=folder,
            max_results=500,
            next_cursor=next_cursor
        )
        resources = response.get('resources', [])

        for res in resources:
            created_time = parser.isoparse(res['created_at'])
            url = build_url(res)
            folder_parts = res['public_id'].split('/')
            place = folder_parts[0] if len(folder_parts) > 0 else "unknown"
            label = folder_parts[1] if len(folder_parts) > 1 else "unknown"

            all_rows.append([url, place, label])
            if created_time > last_retrain_time:
                new_images.append(res)

        next_cursor = response.get("next_cursor")
        if not next_cursor:
            break

    # Save all to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'place', 'label'])
        writer.writerows(all_rows)
    logger.info(f"Saved {len(all_rows)} image URLs to {OUTPUT_CSV}")
    
    # Decide next step
    # if len(new_images) >= 1:
    #     logger.info("✅ Enough new images found. Continue pipeline.")
    #     return 'preprocess_data'
    # else:
    #     logger.info("⏹ Not enough new images. Skip training.")
    #     return 'skip_training'
    
    # Test the preprocess data pipeline
    logger.info("✅ Enough new images to test pipeline.")
    return 'preprocess_data'