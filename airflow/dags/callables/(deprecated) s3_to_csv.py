import cloudinary
import cloudinary.api
import csv
import os
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Setup your Cloudinary credentials ---
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# --- Parameters ---
# FOLDERS = ['Decks/', 'Walls/', 'Pavements/']
FOLDERS = ['Users/']
OUTPUT_CSV = 'cloudinary_dataset_users.csv'
INCLUDE_VERSION = True

# --- Helper to build Cloudinary image URL ---
def build_cloudinary_url(resource, include_version=True):
    base = f"https://res.cloudinary.com/{cloudinary.config().cloud_name}/image/upload"
    path = resource['public_id'] + '.' + resource['format']
    if include_version:
        return f"{base}/v{resource['version']}/{path}"
    else:
        return f"{base}/{path}"

# --- Main function ---
def generate_dataset_csv():
    """
    Scans specified folders in a Cloudinary account, retrieves image resources, and generates a CSV file containing image URLs, place, and label information.

    The function iterates through each folder listed in the global FOLDERS variable, paginates through all resources using the Cloudinary API, and constructs a list of rows with the image URL, place, and label. The results are written to a CSV file specified by the global OUTPUT_CSV variable.

    Side Effects:
        - Logs progress and status messages.
        - Writes a CSV file to disk.

    Assumes the following global variables and functions are defined:
        - FOLDERS: List of folder names to scan.
        - cloudinary: Cloudinary API client.
        - build_cloudinary_url: Function to construct the image URL.
        - INCLUDE_VERSION: Boolean or parameter for URL versioning.
        - OUTPUT_CSV: Output path for the CSV file.
    """
    logger.info(f"📡 Starting scan for folders: {FOLDERS}")
    all_rows = []

    for folder in FOLDERS:
        logger.info(f"🔍 Scanning folder: {folder}")
        next_cursor = None
        total_in_folder = 0

        while True:
            response = cloudinary.api.resources(
                type="upload",
                prefix=folder,
                max_results=500,
                next_cursor=next_cursor
            )
            resources = response.get('resources', [])
            total_in_folder += len(resources)

            for i, res in enumerate(resources):
                url = build_cloudinary_url(res, INCLUDE_VERSION)
                folder_parts = res['public_id'].split('/')
                if len(folder_parts) >= 2:
                    place = folder_parts[0]
                    label = folder_parts[1]
                else:
                    place = folder
                    label = 'unknown'
                all_rows.append([url, place, label])

                if i % 50 == 0:
                    logger.info(f"  → Processed {i+1}/{total_in_folder} images in current batch...")

            next_cursor = response.get('next_cursor')
            if not next_cursor:
                break

        logger.info(f"✅ {total_in_folder} images found in '{folder}'")

    # Write CSV
    logger.info(f"📝 Writing CSV: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'place', 'label'])
        for row in all_rows:
            writer.writerow(row)

    logger.info(f"🎉 Done! Dataset CSV saved to: {OUTPUT_CSV}")

# --- Run ---
if __name__ == "__main__":
    generate_dataset_csv()
