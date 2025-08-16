import os
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
from collections import defaultdict

def load_paths(config_path="../paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

# Paths from configuration
VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["video_dir"])
TRAIN_CSV = os.path.join(BASE_DIR, "..", paths["annotation_csv"])
VAL_CSV = os.path.join(BASE_DIR, "..", paths["validation_csv"])
PROBE_CLIPS_DIR = os.path.join(BASE_DIR, "..", paths["probe_clips_dir"])
TRAIN_OUTPUT_DIR = os.path.join(PROBE_CLIPS_DIR, "train")
VAL_OUTPUT_DIR = os.path.join(PROBE_CLIPS_DIR, "val")

# Create output directories
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

NUM_WORKERS = 8
TRIM_START = 902   # seconds (as specified in the request)
TRIM_END = 1798    # seconds (as specified in the request)
TRIM_DURATION = TRIM_END - TRIM_START  # 896 seconds

def is_valid_video_ffprobe(path):
    """Check if video file is valid using ffprobe"""
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

def get_unique_clips_from_csv(csv_path):
    """Extract unique video clips from CSV annotations"""
    print(f"Reading annotations from: {csv_path}")
    
    # Read CSV without headers (format: video_id, timestamp, x1, y1, x2, y2, action_id, person_id)
    df = pd.read_csv(csv_path, header=None, names=[
        'video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id'
    ])
    
    # Get unique video IDs
    unique_clips = df['video_id'].unique()
    print(f"Found {len(unique_clips)} unique clips in {csv_path}")
    
    return unique_clips

def trim_clip(args):
    """Trim a single video clip"""
    clip_id, output_dir = args
    
    # Look for the video file with various extensions
    possible_extensions = ['.mp4', '.mkv', '.webm', '.avi']
    input_path = None
    
    for ext in possible_extensions:
        potential_path = os.path.join(VIDEO_DIR, f"{clip_id}{ext}")
        if os.path.exists(potential_path):
            input_path = potential_path
            break
    
    if not input_path:
        return f"❌ Video not found: {clip_id}"
    
    # Output filename (always .mp4)
    output_filename = f"{clip_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Skip if already exists and is valid
    if os.path.exists(output_path) and is_valid_video_ffprobe(output_path):
        return f"✔ Skipped (already trimmed): {clip_id}"
    
    # FFmpeg command to trim video
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files
        "-hide_banner", "-loglevel", "error",  # Clean logging
        "-ss", str(TRIM_START),                # Start at 902s
        "-t", str(TRIM_DURATION),              # Duration = 896s (902-1798)
        "-i", input_path,                      # Input file
        "-c:v", "h264_nvenc",                  # GPU encoding (fallback to libx264 if not available)
        "-preset", "fast",                     # Fast encoding preset
        "-an",                                 # Remove audio
        output_path
    ]
    
    try:
        # Try with GPU encoding first
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback to CPU encoding if GPU encoding fails
        cmd_cpu = cmd.copy()
        cmd_cpu[cmd_cpu.index("-c:v") + 1] = "libx264"
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            return f"❌ Error trimming {clip_id}: {e}"
    
    # Verify the output file
    if is_valid_video_ffprobe(output_path):
        return f"✅ Trimmed: {clip_id}"
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        return f"⚠️ Corrupted output: {clip_id}"

def process_dataset(csv_path, output_dir, dataset_name):
    """Process a complete dataset (train or val)"""
    print(f"\n🎬 Processing {dataset_name} dataset...")
    
    # Get unique clips from CSV
    unique_clips = get_unique_clips_from_csv(csv_path)
    
    # Filter clips that have corresponding video files
    existing_clips = []
    for clip_id in unique_clips:
        possible_extensions = ['.mp4', '.mkv', '.webm', '.avi']
        for ext in possible_extensions:
            video_path = os.path.join(VIDEO_DIR, f"{clip_id}{ext}")
            if os.path.exists(video_path):
                existing_clips.append(clip_id)
                break
    
    print(f"Found {len(existing_clips)} video files for {dataset_name} dataset")
    
    if not existing_clips:
        print(f"⚠️ No video files found for {dataset_name} dataset")
        return
    
    # Prepare arguments for multiprocessing
    args_list = [(clip_id, output_dir) for clip_id in existing_clips]
    
    # Process clips with multiprocessing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(trim_clip, args_list), 
            total=len(args_list), 
            desc=f"Trimming {dataset_name} clips", 
            unit="clip"
        ))
    
    # Print results summary
    success_count = sum(1 for r in results if r.startswith("✅"))
    skip_count = sum(1 for r in results if r.startswith("✔"))
    error_count = sum(1 for r in results if r.startswith("❌") or r.startswith("⚠️"))
    
    print(f"\n📊 {dataset_name} Results:")
    print(f"   ✅ Successfully trimmed: {success_count}")
    print(f"   ✔ Skipped (already exists): {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # Print first few errors if any
    errors = [r for r in results if r.startswith("❌") or r.startswith("⚠️")]
    if errors:
        print(f"   📝 First few errors:")
        for error in errors[:5]:
            print(f"      {error}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more errors")

def main():
    print("🎯 AVA Probe Clips Sampler")
    print(f"   Trimming range: {TRIM_START}s - {TRIM_END}s (duration: {TRIM_DURATION}s)")
    print(f"   Video source: {VIDEO_DIR}")
    print(f"   Output directory: {PROBE_CLIPS_DIR}")
    print(f"   Workers: {NUM_WORKERS}")
    
    # Check if video directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return
    
    # Check if CSV files exist
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Train CSV not found: {TRAIN_CSV}")
        return
    
    if not os.path.exists(VAL_CSV):
        print(f"❌ Validation CSV not found: {VAL_CSV}")
        return
    
    # Process train dataset
    process_dataset(TRAIN_CSV, TRAIN_OUTPUT_DIR, "train")
    
    # Process validation dataset  
    process_dataset(VAL_CSV, VAL_OUTPUT_DIR, "val")
    
    print("\n🎉 All datasets processed for probe clip sampling!")

if __name__ == "__main__":
    main()
