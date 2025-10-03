#!/usr/bin/env python3
"""
AVA Video Cutter
Cuts each video from its 15th to 30th minute using GPU acceleration.
Based on Facebook's cut_ava_videos.sh but with Python implementation and GPU optimization.
"""

import os
import subprocess
import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

def load_paths(config_path="paths.yaml"):
    """Load paths from configuration file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def cut_video(args):
    """Cut a single video from 15th to 30th minute"""
    input_path, output_path = args
    
    # Skip if output already exists and is valid
    if os.path.exists(output_path) and is_valid_video_ffprobe(output_path):
        return f"✔️ Skipped (already exists): {os.path.basename(output_path)}"
    
    # FFmpeg command to cut video (15th to 30th minute)
    # -ss 900 (start at 900 seconds = 15 minutes)
    # -t 901 (duration 901 seconds ≈ 15 minutes, matching Facebook's script)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files
        "-hide_banner", "-loglevel", "error",  # Clean logging
        "-ss", "900",                          # Start at 900s (15 minutes)
        "-t", "901",                           # Duration 901s (≈15 minutes)
        "-i", input_path,                      # Input file
        "-c:v", "h264_nvenc",                  # GPU encoding
        "-preset", "fast",                     # Fast encoding preset
        "-c:a", "aac",                         # Audio codec (keep audio unlike probe sampler)
        "-b:a", "128k",                        # Audio bitrate
        output_path
    ]
    
    try:
        # Try with GPU encoding first
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback to CPU encoding if GPU encoding fails
        cmd_cpu = cmd.copy()
        cmd_cpu[cmd_cpu.index("-c:v") + 1] = "libx264"
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            return f"❌ Error cutting {os.path.basename(input_path)}: {e}"
    
    # Verify the output file
    if is_valid_video_ffprobe(output_path):
        return f"✅ Cut: {os.path.basename(output_path)}"
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        return f"⚠️ Corrupted output: {os.path.basename(output_path)}"

def main():
    print("🎯 AVA Video Cutter (15th-30th minute)")
    
    # Load configuration
    paths = load_paths()
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, paths["video_dir"])
    output_dir = os.path.join(base_dir, paths["videos_15min_dir"])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Input directory: {video_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.mkv', '.webm', '.avi']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"🎬 Found {len(video_files)} video files")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for video_file in video_files:
        input_path = str(video_file)
        output_filename = video_file.name  # Keep original extension
        output_path = os.path.join(output_dir, output_filename)
        args_list.append((input_path, output_path))
    
    # Process videos with multiprocessing
    num_workers = min(8, len(args_list))  # Use max 8 workers
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(cut_video, args_list), 
            total=len(args_list), 
            desc="Cutting videos", 
            unit="video"
        ))
    
    # Print results summary
    success_count = sum(1 for r in results if r.startswith("✅"))
    skip_count = sum(1 for r in results if r.startswith("✔️"))
    error_count = sum(1 for r in results if r.startswith("❌") or r.startswith("⚠️"))
    
    print(f"\n📊 Cutting Results:")
    print(f"   ✅ Successfully cut: {success_count}")
    print(f"   ✔️ Skipped (already exists): {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # Print first few errors if any
    errors = [r for r in results if r.startswith("❌") or r.startswith("⚠️")]
    if errors:
        print(f"   📝 First few errors:")
        for error in errors[:5]:
            print(f"      {error}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more errors")
    
    if success_count + skip_count > 0:
        print("\n🎉 Video cutting completed!")
        print(f"📁 Cut videos are in: {output_dir}")
    else:
        print("\n⚠️ No videos were processed successfully.")

if __name__ == "__main__":
    main()