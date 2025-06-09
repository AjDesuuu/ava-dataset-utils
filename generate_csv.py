import os
import csv
import subprocess
import yaml

def load_paths(config_path="paths.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

paths = load_paths()
# Paths from YAML
dataset_dir = paths["output_dir"]  # Path to the 5-second clips directory
output_csv = paths.get("clips_csv", "/home/Aaron/datasets/ava/ava_clips_5s_list.csv")  # Output CSV file

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
    clips = []

    # Walk through the directory to find video clips
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(clip_extension):
                clip_path = os.path.join(root, file)
                
                # Check if the video clip is valid (non-corrupt)
                if is_valid_video_ffprobe(clip_path):
                    clips.append([clip_path, 0])  # Append valid clip with class label 0

    # Write the clip paths and class labels (0) to a CSV file with space delimiter
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=" ")
        writer.writerow(["clip_path", "class_label"])  # CSV header
        for clip in clips:
            writer.writerow(clip)

    print(f"CSV file generated: {output_csv}")
    print(f"Total valid clips found: {len(clips)}")

if __name__ == "__main__":
    scan_and_generate_csv(dataset_dir, output_csv)
