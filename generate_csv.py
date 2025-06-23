import os
import csv
import subprocess
import yaml
from tqdm import tqdm
import concurrent.futures 

def load_paths(config_path="paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()
dataset_dir = os.path.join(BASE_DIR, paths["output_dir"])
output_csv = os.path.join(BASE_DIR, paths["clips_csv"])


# Supported video file extension for clips (we assume `.mp4` here)
clip_extension = ".mp4"

def is_valid_video_ffprobe(clip_path):
    """Check if the video file is valid using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", clip_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        frames = int(result.stdout.decode().strip())
        return frames > 0
    except:
        return False

def scan_and_generate_csv(dataset_dir, output_csv):
    """Scan the dataset directory and generate a CSV with clip paths and dummy class labels."""
    # Gather all candidate files first
    candidate_clips = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(clip_extension):
                candidate_clips.append(os.path.join(root, file))

    # Parallel validation
    valid_clips = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(is_valid_video_ffprobe, candidate_clips),
            total=len(candidate_clips),
            desc="Validating clips",
            unit="clip"
        ))
    for clip_path, is_valid in zip(candidate_clips, results):
        if is_valid:
            valid_clips.append([clip_path, 0])

    # Write the clip paths and class labels (0) to a CSV file with space delimiter
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=" ")
        writer.writerow(["clip_path", "class_label"])  # CSV header
        for clip in valid_clips:
            writer.writerow(clip)

    print(f"CSV file generated: {output_csv}")
    print(f"Total valid clips found: {len(valid_clips)}")

if __name__ == "__main__":
    scan_and_generate_csv(dataset_dir, output_csv)
