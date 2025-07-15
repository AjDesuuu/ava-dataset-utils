#!/usr/bin/env python3
import os
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import yaml

def load_paths(config_path="paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

TRIMMED_VIDEO_DIR = os.path.join(BASE_DIR, paths["trimmed_video_dir"])

def check_video_detailed(video_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode != 0:
            return {"file": os.path.basename(video_path), "status": "ERROR", "error": "ffprobe failed"}

        data = json.loads(result.stdout.decode())
        video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            return {"file": os.path.basename(video_path), "status": "ERROR", "error": "No video streams found"}
        
        stream = video_streams[0]
        format_info = data.get("format", {})

        duration = float(format_info.get("duration", 0))
        nb_frames = int(stream.get("nb_frames", 0))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")
        fps = eval(stream.get("r_frame_rate", "0/1")) if stream.get("r_frame_rate") else 0
        size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)

        issues = []
        if duration < 1: issues.append("Very short duration")
        if nb_frames < 10: issues.append("Very few frames")
        if width < 100 or height < 100: issues.append("Low resolution")
        if fps < 1: issues.append("Low FPS")

        status = "ERROR" if issues else "OK"
        
        return {
            "file": os.path.basename(video_path),
            "status": status,
            "duration": duration,
            "frames": nb_frames,
            "resolution": f"{width}x{height}",
            "codec": codec,
            "fps": round(fps, 2),
            "size_mb": size_mb,
            "issues": issues or None
        }

    except subprocess.TimeoutExpired:
        return {"file": os.path.basename(video_path), "status": "ERROR", "error": "ffprobe timeout"}
    except Exception as e:
        return {"file": os.path.basename(video_path), "status": "ERROR", "error": str(e)}

def main():
    print(f"Checking videos in: {TRIMMED_VIDEO_DIR}")
    if not os.path.exists(TRIMMED_VIDEO_DIR):
        print(f"❌ Directory not found: {TRIMMED_VIDEO_DIR}")
        return

    video_files = [
        os.path.join(TRIMMED_VIDEO_DIR, f) 
        for f in os.listdir(TRIMMED_VIDEO_DIR) 
        if f.lower().endswith(".mp4")
    ]

    if not video_files:
        print("❌ No video files found.")
        return

    print(f"Found {len(video_files)} video files.\n")

    results = []
    error_files = []
    invalid_frame_files = []
    total_size = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(check_video_detailed, f): f for f in video_files}
        for future in tqdm(as_completed(future_to_file), total=len(video_files), desc="Checking videos"):
            result = future.result()
            results.append(result)

            if result["status"] == "ERROR":
                error_files.append(result)
            elif result.get("frames") != 27030:
                invalid_frame_files.append(result)

            if result.get("status") == "OK":
                total_size += result.get("size_mb", 0)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total videos checked: {len(results)}")
    print(f"✅ Valid: {len(results) - len(error_files)}")
    print(f"❌ Errors: {len(error_files)}")
    print(f"⚠️ Not 27030 frames: {len(invalid_frame_files)}")
    print(f"📦 Total size: {total_size:.2f} MB")

    if invalid_frame_files:
        print("\n⚠️ Files with incorrect frame count:")
        for vid in invalid_frame_files:
            print(f" - {vid['file']}: {vid['frames']} frames")

    # Save invalid frame count list
    if invalid_frame_files:
        with open("invalid_frame_count_videos.txt", "w") as f:
            f.write("Videos with frames not equal to 27030:\n")
            f.write("="*40 + "\n")
            for vid in invalid_frame_files:
                f.write(f"{vid['file']} - {vid['frames']} frames\n")
        print("📝 Saved to: invalid_frame_count_videos.txt")

    # Save error list
    if error_files:
        with open("error_videos.txt", "w") as f:
            f.write("Videos with errors:\n")
            f.write("="*40 + "\n")
            for vid in error_files:
                f.write(f"{vid['file']} - {vid.get('error', 'Unknown error')}\n")
        print("📝 Saved to: error_videos.txt")

if __name__ == "__main__":
    main()
