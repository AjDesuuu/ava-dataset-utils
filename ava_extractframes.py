#!/usr/bin/env python3
"""
AVA Frame Extractor
Extracts frames from cut videos using GPU acceleration and creates frame lists.
Based on Facebook's extract_ava_frames.sh but with Python implementation and GPU optimization.
"""

import os
import subprocess
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

def load_paths(config_path="paths.yaml"):
    """Load paths from configuration file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_frames_from_video(args):
    """Extract frames from a single video"""
    video_path, frames_dir, video_name_no_ext = args
    
    # Create output directory for this video
    video_frames_dir = os.path.join(frames_dir, video_name_no_ext)
    os.makedirs(video_frames_dir, exist_ok=True)
    
    # Check if frames already exist
    existing_frames = list(Path(video_frames_dir).glob("*.jpg"))
    if existing_frames:
        return f"✔️ Skipped (frames exist): {video_name_no_ext} ({len(existing_frames)} frames)"
    
    # Output pattern for frames
    output_pattern = os.path.join(video_frames_dir, f"{video_name_no_ext}_%06d.jpg")
    
    # FFmpeg command to extract frames
    # -r 30: Extract at 30 FPS
    # -q:v 1: Highest quality JPEG (1-31, lower is better)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files
        "-hide_banner", "-loglevel", "error",  # Clean logging
        "-hwaccel", "cuda",                    # Use CUDA hardware acceleration
        "-i", video_path,                      # Input video
        "-r", "30",                            # Extract at 30 FPS
        "-q:v", "1",                           # Highest quality JPEG
        "-f", "image2",                        # Image format
        output_pattern
    ]
    
    try:
        # Try with GPU acceleration first
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback to CPU processing if GPU acceleration fails
        cmd_cpu = cmd.copy()
        cmd_cpu.remove("-hwaccel")
        cmd_cpu.remove("cuda")
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Clean up any partial extraction
            if os.path.exists(video_frames_dir):
                import shutil
                shutil.rmtree(video_frames_dir)
            return f"❌ Error extracting {video_name_no_ext}: {e}"
    
    # Count extracted frames
    extracted_frames = list(Path(video_frames_dir).glob("*.jpg"))
    
    if extracted_frames:
        return f"✅ Extracted: {video_name_no_ext} ({len(extracted_frames)} frames)"
    else:
        # Clean up empty directory
        if os.path.exists(video_frames_dir):
            os.rmdir(video_frames_dir)
        return f"⚠️ No frames extracted: {video_name_no_ext}"

def create_frame_lists(frames_dir, frame_lists_dir, annotations_dir):
    """Create frame lists (train.csv and val.csv) based on annotations"""
    print("\n📋 Creating frame lists...")
    
    # Load train and validation annotations
    train_csv = os.path.join(annotations_dir, "ava_train_v2.2.csv")
    val_csv = os.path.join(annotations_dir, "ava_val_v2.2.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print("⚠️ Annotation files not found. Skipping frame list creation.")
        print("   Run ava_annotation.py first to download annotations.")
        return
    
    # Read annotations
    train_df = pd.read_csv(train_csv, header=None, names=[
        'video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id'
    ])
    val_df = pd.read_csv(val_csv, header=None, names=[
        'video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id'
    ])
    
    # Get unique video IDs
    train_videos = set(train_df['video_id'].unique())
    val_videos = set(val_df['video_id'].unique())
    
    # Find all extracted frame directories
    frame_dirs = [d for d in Path(frames_dir).iterdir() if d.is_dir()]
    
    # Create frame lists
    train_frames = []
    val_frames = []
    
    for frame_dir in frame_dirs:
        video_name = frame_dir.name
        
        # Get all frame files for this video
        frame_files = sorted(frame_dir.glob("*.jpg"))
        
        if not frame_files:
            continue
        
        # Determine if this video belongs to train or val set
        if video_name in train_videos:
            for frame_file in frame_files:
                # Format: video_frames_path, label (0 for frames without specific labels)
                relative_path = os.path.join(video_name, frame_file.name)
                train_frames.append([relative_path, 0])
        elif video_name in val_videos:
            for frame_file in frame_files:
                relative_path = os.path.join(video_name, frame_file.name)
                val_frames.append([relative_path, 0])
    
    # Save frame lists
    train_list_path = os.path.join(frame_lists_dir, "train.csv")
    val_list_path = os.path.join(frame_lists_dir, "val.csv")
    
    if train_frames:
        train_list_df = pd.DataFrame(train_frames, columns=['frame_path', 'label'])
        train_list_df.to_csv(train_list_path, index=False)
        print(f"✅ Created train.csv with {len(train_frames)} frames")
    
    if val_frames:
        val_list_df = pd.DataFrame(val_frames, columns=['frame_path', 'label'])
        val_list_df.to_csv(val_list_path, index=False)
        print(f"✅ Created val.csv with {len(val_frames)} frames")

def main():
    print("🎯 AVA Frame Extractor")
    
    # Load configuration
    paths = load_paths()
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_15min_dir = os.path.join(base_dir, paths["videos_15min_dir"])
    frames_dir = os.path.join(base_dir, paths["frame_dir"])
    frame_lists_dir = os.path.join(base_dir, paths["frame_lists_dir"])
    annotations_dir = os.path.join(base_dir, paths["annotations_dir"])
    
    # Create output directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(frame_lists_dir, exist_ok=True)
    
    print(f"📁 Input directory: {videos_15min_dir}")
    print(f"📁 Frames output: {frames_dir}")
    print(f"📁 Frame lists output: {frame_lists_dir}")
    
    # Check if input directory exists
    if not os.path.exists(videos_15min_dir):
        print(f"❌ 15-minute videos directory not found: {videos_15min_dir}")
        print("   Run ava_cutvids.py first to create 15-minute clips.")
        return
    
    # Find all video files in 15min directory
    video_extensions = ['.mp4', '.mkv', '.webm', '.avi']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(videos_15min_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"❌ No video files found in {videos_15min_dir}")
        return
    
    print(f"🎬 Found {len(video_files)} video files")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for video_file in video_files:
        video_path = str(video_file)
        
        # Extract video name without extension
        if video_file.suffix == '.webm':
            video_name_no_ext = video_file.stem
        else:
            video_name_no_ext = video_file.stem
        
        args_list.append((video_path, frames_dir, video_name_no_ext))
    
    # Extract frames with multiprocessing
    num_workers = min(4, len(args_list))  # Use max 4 workers for frame extraction (I/O intensive)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(extract_frames_from_video, args_list), 
            total=len(args_list), 
            desc="Extracting frames", 
            unit="video"
        ))
    
    # Print results summary
    success_count = sum(1 for r in results if r.startswith("✅"))
    skip_count = sum(1 for r in results if r.startswith("✔️"))
    error_count = sum(1 for r in results if r.startswith("❌") or r.startswith("⚠️"))
    
    print(f"\n📊 Frame Extraction Results:")
    print(f"   ✅ Successfully extracted: {success_count}")
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
    
    # Create frame lists
    if success_count + skip_count > 0:
        create_frame_lists(frames_dir, frame_lists_dir, annotations_dir)
        
        print("\n🎉 Frame extraction completed!")
        print(f"📁 Frames are in: {frames_dir}")
        print(f"📁 Frame lists are in: {frame_lists_dir}")
        
        # Print directory structure summary
        frame_dirs = [d for d in Path(frames_dir).iterdir() if d.is_dir()]
        total_frames = sum(len(list(d.glob("*.jpg"))) for d in frame_dirs)
        print(f"\n📊 Final Structure Summary:")
        print(f"   🎬 Video directories: {len(frame_dirs)}")
        print(f"   🖼️ Total frames extracted: {total_frames}")
    else:
        print("\n⚠️ No frames were extracted successfully.")

if __name__ == "__main__":
    main()