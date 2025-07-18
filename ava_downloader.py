import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import yaml

def load_paths(config_path="paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()
# Settings
split = "trainval"
file_list_path = os.path.join(BASE_DIR, paths["file_list_test"] if split == "test" else paths["file_list_trainval"])
output_dir = os.path.join(BASE_DIR, paths["video_dir"].replace("trainval", split))
base_url = f'{paths["base_url"]}/{split}/'
max_threads = 4
max_retries = 3

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read filenames
with open(file_list_path, "r") as f:
    file_names = [line.strip() for line in f if line.strip()]

def download_file(file_name):
    url = f"{base_url}{file_name}"
    output_path = os.path.join(output_dir, file_name)

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                # Skip if file exists and is the correct size
                if os.path.exists(output_path) and os.path.getsize(output_path) == total_size:
                    return f"✔ Skipped (already complete): {file_name}"

                # If exists but wrong size, delete it
                if os.path.exists(output_path):
                    os.remove(output_path)

                # Download with progress bar
                with open(output_path, "wb") as f_out, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file_name,
                    position=0,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
                            bar.update(len(chunk))

                # Verify after writing
                actual_size = os.path.getsize(output_path)
                if actual_size != total_size:
                    raise Exception(f"Incomplete download: expected {total_size}, got {actual_size}")

                return f"✅ Downloaded: {file_name}"

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                print(f"⚠️ Retry {attempt}/{max_retries} for {file_name}: {e}")
            else:
                return f"❌ Failed after {max_retries} attempts: {file_name}"

# Run downloads in parallel
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(download_file, f) for f in file_names]
    for future in as_completed(futures):
        print(future.result())
