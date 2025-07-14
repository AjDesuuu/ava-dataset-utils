import os
import zipfile
import gdown

# Base path (you are inside datasets/ava-dataset-utils/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
zip_path = os.path.join(base_dir, "semantic_masks.zip")
output_dir = os.path.join(base_dir, "ava")
os.makedirs(output_dir, exist_ok=True)

# Google Drive file ID
file_id = "1OrvgjqdQgVswSg-MlQAQCiKpc1LIYueq"
gdown_url = f"https://drive.google.com/uc?id={file_id}"

# Step 1: Download
print("📥 Downloading ZIP file...")
gdown.download(gdown_url, zip_path, quiet=False)

# Step 2: Unzip into semantic_masks/
print("📦 Extracting to 'semantic_masks/'...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(output_dir)

# Step 3: Remove zip
os.remove(zip_path)
print(f"✅ Done! Extracted to: {output_dir}")
