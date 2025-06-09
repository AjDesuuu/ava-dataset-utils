import csv
import os
import av
import subprocess
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from statistics import median

# --- CONFIGURATION ---
VIDEO_DIR = "/home/Aaron/datasets/ava/videos/trainval"
ANNOTATION_CSV = "/home/Aaron/datasets/ava/annotations/ava_train_v2.2.csv"
OUTPUT_DIR = "/home/Aaron/datasets/ava/sampled_clips"
CLUSTER_WINDOW = 5      # seconds between timestamps to consider them part of the same cluster
MIN_DURATION = 7        # minimum clip length
BUFFER = 1.0            # buffer to ensure scenes aren't cut too tight
MAX_DURATION = 15       # max duration (e.g., based on the average duration of SSV2 clips)
NUM_WORKERS = 8         # parallel processes

# --- SETUP ---
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

def cluster_timestamps_with_duration(timestamps, window=5, min_duration=3, max_duration=10, buffer=1.0):
    timestamps = sorted(set(timestamps))
    clusters = []
    current = []

    for t in timestamps:
        if not current:
            current = [t]
        elif t - current[-1] < window:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    if current:
        clusters.append(current)

    results = []
    for cluster in clusters:
        start = min(cluster)
        end = max(cluster)
        duration = max(min_duration, end - start + buffer)

        # Ensure the duration doesn't exceed the max allowed duration
        duration = min(duration, max_duration)

        counter = Counter(cluster)
        if counter:
            mode_ts, freq = counter.most_common(1)[0]
            if freq > 1:
                center_ts = int(mode_ts)
            else:
                center_ts = int(median(cluster))

        results.append((center_ts, duration))

    return results

# --- MAIN ---
video_timestamps = defaultdict(list)
with open(ANNOTATION_CSV, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        video_id, timestamp = row[0], float(row[1])
        video_timestamps[video_id].append(timestamp)

tasks = []
seen = set()
for video_id, timestamps in video_timestamps.items():
    clustered = cluster_timestamps_with_duration(timestamps, window=CLUSTER_WINDOW, min_duration=MIN_DURATION, max_duration=MAX_DURATION, buffer=BUFFER)
    for center_ts, duration in clustered:
        key = (video_id, center_ts, duration)
        if key not in seen:
            seen.add(key)
            tasks.append(key)

# Run extraction with parallel workers and track progress
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for result in tqdm(executor.map(extract_clip, tasks), total=len(tasks), desc="Processing clips", unit="clip"):
        print(result)

print("🎉 All dynamic clips processed.")
