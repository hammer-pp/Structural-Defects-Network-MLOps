import os
import csv
import cloudinary
import cloudinary.api
import logging
import pandas as pd
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

    OUTPUT_CSV = '/opt/airflow/cloudinary_dataset_users.csv'
    folders = ["Users/Cracked/", "Users/Non-cracked/"]
    include_version = True

    def build_url(resource):
        base = f"https://res.cloudinary.com/{cloudinary.config().cloud_name}/image/upload"
        path = resource['public_id'] + '.' + resource['format']
        return f"{base}/v{resource['version']}/{path}" if include_version else f"{base}/{path}"

    # Load previous retraining timestamp
    last_retrain_str = Variable.get("last_retrain_time", default_var="2025-01-01T00:00:00")
    last_retrain_time = datetime.fromisoformat(last_retrain_str).replace(tzinfo=None)
    logger.info(f"Last retrain: {last_retrain_time}")

    # Load existing CSV if present
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        logged_urls = set(df_existing['url'])
    else:
        df_existing = pd.DataFrame(columns=['url', 'place', 'label', 'created_at'])
        logged_urls = set()

    new_rows = []
    new_image_count = 0

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

                if url in logged_urls:
                    continue

                folder_parts = res['public_id'].split('/')
                place = folder_parts[0] if len(folder_parts) > 0 else "unknown"
                label = folder_parts[1] if len(folder_parts) > 1 else "unknown"

                new_rows.append([url, place, label, created_time.isoformat()])

                if created_time > last_retrain_time:
                    new_image_count += 1

            next_cursor = response.get("next_cursor")
            if not next_cursor:
                break

    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=['url', 'place', 'label', 'created_at'])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"📄 Appended {len(new_rows)} new records to {OUTPUT_CSV}")
    else:
        logger.info("📄 No new records to append to CSV")

    # if new_image_count >= 0:
    #     logger.info(f"✅ Found {new_image_count} new images since last retrain → preprocessing.")
    #     return 'preprocess_data'
    # else:
    #     logger.info(f"⏹ Only {new_image_count} new images found → skipping.")
    #     return 'skip_training'
    return 'preprocess_data'
