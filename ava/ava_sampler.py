import csv
import os
import av
import subprocess
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from statistics import median
import yaml

def load_paths(config_path="../paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["trimmed_video_dir"])
ANNOTATION_CSV = os.path.join(BASE_DIR, "..", paths["annotation_csv"])
OUTPUT_DIR = os.path.join(BASE_DIR, "..", paths["output_dir"])
NUM_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_video_file(video_id):
    path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    return path if os.path.exists(path) else None

def is_valid_video_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        frames = int(result.stdout.decode().strip())
        return frames > 0
    except:
        return False

def extract_clip(task):
    video_id, timestamp, duration = task
    video_path = find_video_file(video_id)
    if not video_path:
        return f"⚠️ Missing file for {video_id}"

    output_filename = f"{video_id}_{int(timestamp)}s_{int(duration)}s.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if os.path.exists(output_path) and is_valid_video_ffprobe(output_path):
        return f"✔ Skipped (already exists): {output_filename}"

    start_time = max(0, timestamp - duration / 2)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", video_path,
        "-c", "copy",    # ✅ no re-encoding
        "-an",           # ✅ remove audio (optional, if not already stripped)
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        if is_valid_video_ffprobe(output_path):
            return f"✅ Trimmed: {output_filename}"
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return f"⚠️ Corrupted output: {output_filename}"
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return f"❌ Error trimming {video_id}_{timestamp}: {e}"

def get_video_duration(video_path):
    """Return the duration of the video in seconds."""
    try:
        with av.open(video_path) as container:
            return float(container.duration / av.time_base)
    except Exception:
        return None

def get_relevant_video_ids(annotation_csv):
    """Parse the annotation file and return a set of relevant video IDs."""
    relevant_ids = set()
    with open(annotation_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            video_id = row[0]
            relevant_ids.add(video_id)
    return relevant_ids

def sliding_window_tasks(video_dir, relevant_ids, window_size=15, stride=15):
    """
    Generate tasks for sliding window clip extraction.
    Only for relevant 15-minute AVA video segments.
    """
    tasks = []
    for file in os.listdir(video_dir):
        if not file.endswith(('.mp4', '.mkv', '.webm')):
            continue
        video_id, ext = os.path.splitext(file)
        video_path = os.path.join(video_dir, file)
        duration = get_video_duration(video_path)
        if duration is None:
            continue
        start = 0
        while start + window_size <= duration:
            center_ts = int(start + window_size // 2)
            tasks.append((video_id, center_ts, window_size))
            start += stride
    return tasks

# --- MAIN ---
relevant_ids = get_relevant_video_ids(ANNOTATION_CSV)
tasks = sliding_window_tasks(VIDEO_DIR, relevant_ids, window_size=15, stride=15)

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for result in tqdm(executor.map(extract_clip, tasks), total=len(tasks), desc="Processing clips", unit="clip"):
        print(result)

print("🎉 All sliding window clips processed.")