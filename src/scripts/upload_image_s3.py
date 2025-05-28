import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from dotenv import load_dotenv

load_dotenv()

# --- Cloudinary config ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

file_path = "../../Dataset/Users/Cracked/326c4c8a42044a68988c07e46e2fbb34.jpg"
relative_path = "Users/Cracked"
filename = os.path.splitext(os.path.basename(file_path))[0]  # e.g. 'sample'
# --- Upload the image ---
upload_response = cloudinary.uploader.upload(
            file_path,
            folder=relative_path,
            public_id=filename,
            unique_filename=False,  # Keep filename
            overwrite=False,        # Skip if already exists
            resource_type="image"
        )
print(f"✅ Uploaded to: {upload_response['secure_url']}")

# --- Fetch resources to confirm it's visible ---
resources = cloudinary.api.resources(type="upload", prefix="Users/", max_results=10)
print(f"\n📂 Found {len(resources['resources'])} images under 'Users/'")
for res in resources['resources']:
    print(f"- {res['public_id']} | created_at: {res['created_at']}")