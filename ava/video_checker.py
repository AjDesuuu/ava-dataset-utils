#!/usr/bin/env python3
"""
Video Checker Script for Trimmed Videos
Checks if videos in the trimmed directory are playable and valid.
"""

import os
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import yaml

def load_paths(config_path="../paths.yaml"):
    """Load paths from configuration file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

TRIMMED_VIDEO_DIR = os.path.join(BASE_DIR, "..", paths["trimmed_video_dir"])

def check_video_detailed(video_path):
    """
    Comprehensive video check using ffprobe
    Returns detailed information about the video
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode != 0:
            return {
                "file": os.path.basename(video_path),
                "status": "ERROR",
                "error": "ffprobe failed",
                "details": result.stderr.decode()
            }
        
        data = json.loads(result.stdout.decode())
        
        # Check if we have video streams
        video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            return {
                "file": os.path.basename(video_path),
                "status": "ERROR",
                "error": "No video streams found"
            }
        
        video_stream = video_streams[0]
        format_info = data.get("format", {})
        
        # Extract key information
        duration = float(format_info.get("duration", 0))
        nb_frames = int(video_stream.get("nb_frames", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "unknown")
        fps = eval(video_stream.get("r_frame_rate", "0/1")) if video_stream.get("r_frame_rate") else 0
        
        # Check for common issues
        issues = []
        if duration < 1:
            issues.append("Very short duration")
        if nb_frames < 10:
            issues.append("Very few frames")
        if width < 100 or height < 100:
            issues.append("Very low resolution")
        if fps < 1:
            issues.append("Very low FPS")
        
        status = "ERROR" if issues else "OK"
        
        return {
            "file": os.path.basename(video_path),
            "status": status,
            "duration": duration,
            "frames": nb_frames,
            "resolution": f"{width}x{height}",
            "codec": codec,
            "fps": round(fps, 2),
            "size_mb": round(os.path.getsize(video_path) / (1024*1024), 2),
            "issues": issues if issues else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "file": os.path.basename(video_path),
            "status": "ERROR",
            "error": "ffprobe timeout"
        }
    except Exception as e:
        return {
            "file": os.path.basename(video_path),
            "status": "ERROR",
            "error": str(e)
        }

def check_video_simple(video_path):
    """
    Simple video check - just verify if ffprobe can read it
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if result.returncode != 0:
            return {
                "file": os.path.basename(video_path),
                "status": "ERROR",
                "error": "Cannot read video"
            }
        
        frames = int(result.stdout.decode().strip())
        if frames > 0:
            return {
                "file": os.path.basename(video_path),
                "status": "OK",
                "frames": frames
            }
        else:
            return {
                "file": os.path.basename(video_path),
                "status": "ERROR",
                "error": "No frames found"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "file": os.path.basename(video_path),
            "status": "ERROR",
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "file": os.path.basename(video_path),
            "status": "ERROR",
            "error": str(e)
        }

def main():
    print(f"Checking videos in: {TRIMMED_VIDEO_DIR}")
    
    if not os.path.exists(TRIMMED_VIDEO_DIR):
        print(f"❌ Directory not found: {TRIMMED_VIDEO_DIR}")
        return
    
    # Find all video files
    video_extensions = ('.mp4', '.mkv', '.webm', '.avi', '.mov')
    video_files = [
        os.path.join(TRIMMED_VIDEO_DIR, f) 
        for f in os.listdir(TRIMMED_VIDEO_DIR) 
        if f.lower().endswith(video_extensions)
    ]
    
    if not video_files:
        print("❌ No video files found in the directory")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Ask user for check type
    print("\nChoose check type:")
    print("1. Simple check (fast)")
    print("2. Detailed check (slower, more info)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    check_func = check_video_simple if choice == "1" else check_video_detailed
    
    # Process videos
    results = []
    error_files = []
    
    print(f"\nChecking {len(video_files)} videos...")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(check_func, video_file): video_file for video_file in video_files}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(video_files), desc="Checking videos"):
            result = future.result()
            results.append(result)
            
            if result["status"] == "ERROR":
                error_files.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos checked: {len(results)}")
    print(f"✅ Valid videos: {len(results) - len(error_files)}")
    print(f"❌ Error videos: {len(error_files)}")
    
    if error_files:
        print(f"\n{'='*60}")
        print("ERROR VIDEOS")
        print(f"{'='*60}")
        for error in error_files:
            print(f"❌ {error['file']}")
            if 'error' in error:
                print(f"   Error: {error['error']}")
            if 'issues' in error and error['issues']:
                print(f"   Issues: {', '.join(error['issues'])}")
            if 'details' in error:
                print(f"   Details: {error['details'][:100]}...")
            print()
    
    # If detailed check, show some stats
    if choice == "2" and results:
        valid_results = [r for r in results if r["status"] == "OK"]
        if valid_results:
            print(f"\n{'='*60}")
            print("STATISTICS (Valid Videos Only)")
            print(f"{'='*60}")
            
            durations = [r.get("duration", 0) for r in valid_results if r.get("duration")]
            sizes = [r.get("size_mb", 0) for r in valid_results if r.get("size_mb")]
            
            if durations:
                print(f"Duration - Min: {min(durations):.1f}s, Max: {max(durations):.1f}s, Avg: {sum(durations)/len(durations):.1f}s")
            if sizes:
                print(f"File size - Min: {min(sizes):.1f}MB, Max: {max(sizes):.1f}MB, Avg: {sum(sizes)/len(sizes):.1f}MB")
    
    # Save error list to file
    if error_files:
        error_file_path = os.path.join(BASE_DIR, "error_videos.txt")
        with open(error_file_path, "w") as f:
            f.write("Error Videos List\n")
            f.write("================\n\n")
            for error in error_files:
                f.write(f"{error['file']}\n")
                if 'error' in error:
                    f.write(f"  Error: {error['error']}\n")
                f.write("\n")
        print(f"\n📄 Error list saved to: {error_file_path}")

if __name__ == "__main__":
    main()
