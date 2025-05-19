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
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )

    OUTPUT_CSV = 'cloudinary_dataset_users.csv'
    folders = ["Users/Cracked/", "Users/Non-cracked/"]
    include_version = True

    def build_url(resource):
        base = f"https://res.cloudinary.com/{cloudinary.config().cloud_name}/image/upload"
        path = resource['public_id'] + '.' + resource['format']
        if include_version:
            return f"{base}/v{resource['version']}/{path}"
        else:
            return f"{base}/{path}"

    last_retrain_str = Variable.get("last_retrain_time", default_var="2025-01-01T00:00:00")
    last_retrain_time = datetime.fromisoformat(last_retrain_str).replace(tzinfo=None)
    logger.info(f"Last retrain: {last_retrain_time}")

    all_rows = []
    new_images = []

    for folder in folders:
        logger.info(f"Scanning folder: {folder}")
        next_cursor = None
        while True:
            response = cloudinary.api.resources(
                type="upload",
                prefix=folder,
                max_results=500,
                next_cursor=next_cursor
            )
            resources = response.get('resources', [])
            logger.info(f"Found {len(resources)} resources in {folder}")

            for res in resources:
                created_time = parser.isoparse(res['created_at']).replace(tzinfo=None)
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

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'place', 'label'])
        writer.writerows(all_rows)

    logger.info(f"📄 Saved {len(all_rows)} total image records to {OUTPUT_CSV}")

    if len(new_images) >= 1:
        logger.info("✅ New images found → continuing pipeline.")
        return 'preprocess_data'
    else:
        logger.info("⏹ No new images → skipping training.")
        return 'skip_training'
