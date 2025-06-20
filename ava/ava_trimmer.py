import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml

def load_paths(config_path="../paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["video_dir"])
TRIMMED_VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["trimmed_video_dir"])
os.makedirs(TRIMMED_VIDEO_DIR, exist_ok=True)

NUM_WORKERS = 8
TRIM_START = 900   # seconds
TRIM_DURATION = 900  # seconds (15 minutes)

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

def trim_video(file):
    input_path = os.path.join(VIDEO_DIR, file)
    output_filename = os.path.splitext(file)[0] + ".mp4"  # Always .mp4
    output_path = os.path.join(TRIMMED_VIDEO_DIR, output_filename)
    if os.path.exists(output_path) and is_valid_video_ffprobe(output_path):
        return f"✔ Skipped (already trimmed): {file}"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner", "-loglevel", "error",  # ✅ clean logging
        "-ss", str(TRIM_START),                          # start at 900s
        "-t", str(TRIM_DURATION+1),                           # duration = 15m 1s
        "-i", input_path,
        "-r", "30",                            # enforce consistent FPS
        "-c:v", "h264_nvenc",                  # ✅ GPU encoding
        "-preset", "fast",                     # fast NVENC preset
        "-b:v", "2M",                          # bitrate target
        "-an",                                 # remove audio
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        if is_valid_video_ffprobe(output_path):
            return f"✅ Trimmed: {file}"
        else:
            # if os.path.exists(output_path):
            #     os.remove(output_path)
            return f"⚠️ Corrupted output: {file}"
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return f"❌ Error trimming {file}: {e}"

if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mkv', '.webm'))]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in tqdm(executor.map(trim_video, video_files), total=len(video_files), desc="Trimming videos", unit="video"):
            print(result)
    print("🎉 All videos processed for trimming.")