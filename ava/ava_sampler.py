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

VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["video_dir"])
ANNOTATION_CSV = os.path.join(BASE_DIR, "..", paths["annotation_csv"])
OUTPUT_DIR = os.path.join(BASE_DIR, "..", paths["output_dir"])
NUM_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_video_file(video_id):
    for ext in [".mp4", ".mkv", ".webm"]:
        path = os.path.join(VIDEO_DIR, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None

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
    end_time = start_time + duration

    try:
        input_container = av.open(video_path)
        video_stream = input_container.streams.video[0]
        input_container.seek(int(start_time / video_stream.time_base), any_frame=False, backward=True, stream=video_stream)

        output_container = av.open(output_path, mode='w')
        out_stream = output_container.add_stream("libx264", rate=video_stream.average_rate)
        out_stream.width = video_stream.codec_context.width
        out_stream.height = video_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "28"}

        frames_written = 0
        for frame in input_container.decode(video=0):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * video_stream.time_base)
            if frame_time < start_time:
                continue
            if frame_time > end_time:
                break
            packet = out_stream.encode(frame)
            if packet:
                output_container.mux(packet)
                frames_written += 1

        for packet in out_stream.encode():
            output_container.mux(packet)

        output_container.close()

        if frames_written > 0 and is_valid_video_ffprobe(output_path):
            return f"✅ Extracted: {output_filename}"
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return f"⚠️ Corrupted or failed: {output_filename}"

    except Exception as e:
        return f"❌ Error for {video_id}_{timestamp}: {e}"

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
        if video_id not in relevant_ids:
            continue
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