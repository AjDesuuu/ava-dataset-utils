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

SAMPLED_CLIPS_DIR = os.path.join(BASE_DIR, paths["output_dir"])

def check_sampled_clip_detailed(video_path):
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
        
        # Check if it was GPU or CPU encoded based on encoder metadata
        encoder = format_info.get("tags", {}).get("encoder", "unknown")
        encoding_method = "GPU" if "nvenc" in encoder.lower() else ("CPU" if "x264" in encoder.lower() else "Unknown")

        issues = []
        # Expected: 15-second clips at 30fps = 450 frames
        if nb_frames != 450:
            issues.append(f"Wrong frame count: {nb_frames} (expected 450)")
        if abs(duration - 15.0) > 0.5:  # Allow 0.5s tolerance
            issues.append(f"Wrong duration: {duration:.2f}s (expected ~15s)")
        if abs(fps - 30.0) > 1.0:  # Allow 1fps tolerance
            issues.append(f"Wrong FPS: {fps:.2f} (expected ~30)")
        if width < 100 or height < 100:
            issues.append("Low resolution")
        if codec not in ["h264", "libx264"]:
            issues.append(f"Unexpected codec: {codec}")

        status = "ERROR" if issues else "OK"
        
        return {
            "file": os.path.basename(video_path),
            "status": status,
            "duration": round(duration, 2),
            "frames": nb_frames,
            "resolution": f"{width}x{height}",
            "codec": codec,
            "fps": round(fps, 2),
            "size_mb": size_mb,
            "encoding_method": encoding_method,
            "encoder": encoder,
            "issues": issues or None
        }

    except subprocess.TimeoutExpired:
        return {"file": os.path.basename(video_path), "status": "ERROR", "error": "ffprobe timeout"}
    except Exception as e:
        return {"file": os.path.basename(video_path), "status": "ERROR", "error": str(e)}

def main():
    print(f"Checking sampled clips in: {SAMPLED_CLIPS_DIR}")
    if not os.path.exists(SAMPLED_CLIPS_DIR):
        print(f"❌ Directory not found: {SAMPLED_CLIPS_DIR}")
        return

    video_files = [
        os.path.join(SAMPLED_CLIPS_DIR, f) 
        for f in os.listdir(SAMPLED_CLIPS_DIR) 
        if f.lower().endswith(".mp4")
    ]

    if not video_files:
        print("❌ No video files found.")
        return

    print(f"Found {len(video_files)} sampled clip files.\n")

    results = []
    error_files = []
    wrong_frame_files = []
    wrong_duration_files = []
    gpu_encoded = 0
    cpu_encoded = 0
    total_size = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(check_sampled_clip_detailed, f): f for f in video_files}
        for future in tqdm(as_completed(future_to_file), total=len(video_files), desc="Checking sampled clips"):
            result = future.result()
            results.append(result)

            if result["status"] == "ERROR":
                error_files.append(result)
            else:
                # Count encoding methods
                if result.get("encoding_method") == "GPU":
                    gpu_encoded += 1
                elif result.get("encoding_method") == "CPU":
                    cpu_encoded += 1
                
                # Check for issues
                if result.get("frames") != 450:
                    wrong_frame_files.append(result)
                if abs(result.get("duration", 0) - 15.0) > 0.5:
                    wrong_duration_files.append(result)
                
                total_size += result.get("size_mb", 0)

    # Print detailed summary
    print("\n" + "=" * 70)
    print("SAMPLED CLIPS VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total clips checked: {len(results)}")
    print(f"✅ Valid clips: {len(results) - len(error_files)}")
    print(f"❌ Error clips: {len(error_files)}")
    print(f"⚠️ Wrong frame count: {len(wrong_frame_files)}")
    print(f"⚠️ Wrong duration: {len(wrong_duration_files)}")
    print(f"📦 Total size: {total_size:.2f} MB")
    print(f"📦 Average size per clip: {total_size/len(results):.2f} MB")
    print()
    print("ENCODING METHOD BREAKDOWN:")
    print(f"🚀 GPU encoded (NVENC): {gpu_encoded}")
    print(f"🖥️  CPU encoded (x264): {cpu_encoded}")
    print(f"❓ Unknown method: {len(results) - gpu_encoded - cpu_encoded - len(error_files)}")

    # Show sample of good clips
    good_clips = [r for r in results if r["status"] == "OK"]
    if good_clips:
        print(f"\n✅ SAMPLE OF GOOD CLIPS:")
        for clip in good_clips[:3]:
            print(f"  {clip['file']}: {clip['frames']} frames, {clip['duration']}s, {clip['fps']}fps, {clip['encoding_method']}")

    # Show problematic clips
    if wrong_frame_files:
        print(f"\n⚠️ CLIPS WITH WRONG FRAME COUNT:")
        for clip in wrong_frame_files[:10]:  # Show first 10
            print(f"  {clip['file']}: {clip['frames']} frames (expected 450)")
        if len(wrong_frame_files) > 10:
            print(f"  ... and {len(wrong_frame_files) - 10} more")

    if wrong_duration_files:
        print(f"\n⚠️ CLIPS WITH WRONG DURATION:")
        for clip in wrong_duration_files[:10]:  # Show first 10
            print(f"  {clip['file']}: {clip['duration']}s (expected ~15s)")
        if len(wrong_duration_files) > 10:
            print(f"  ... and {len(wrong_duration_files) - 10} more")

    if error_files:
        print(f"\n❌ ERROR CLIPS:")
        for clip in error_files[:10]:  # Show first 10
            print(f"  {clip['file']}: {clip.get('error', 'Unknown error')}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more")

    # Save detailed report
    report_filename = "sampled_clips_report.txt"
    with open(report_filename, "w") as f:
        f.write("SAMPLED CLIPS VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total clips: {len(results)}\n")
        f.write(f"Valid clips: {len(results) - len(error_files)}\n")
        f.write(f"Error clips: {len(error_files)}\n")
        f.write(f"Wrong frame count: {len(wrong_frame_files)}\n")
        f.write(f"Wrong duration: {len(wrong_duration_files)}\n")
        f.write(f"GPU encoded: {gpu_encoded}\n")
        f.write(f"CPU encoded: {cpu_encoded}\n")
        f.write(f"Total size: {total_size:.2f} MB\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in sorted(results, key=lambda x: x["file"]):
            f.write(f"File: {result['file']}\n")
            f.write(f"  Status: {result['status']}\n")
            if result['status'] != 'ERROR':
                f.write(f"  Frames: {result['frames']} | Duration: {result['duration']}s | FPS: {result['fps']}\n")
                f.write(f"  Resolution: {result['resolution']} | Codec: {result['codec']}\n")
                f.write(f"  Size: {result['size_mb']} MB | Method: {result['encoding_method']}\n")
                f.write(f"  Encoder: {result['encoder']}\n")
                if result['issues']:
                    f.write(f"  Issues: {', '.join(result['issues'])}\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown')}\n")
            f.write("\n")

    print(f"\n📝 Detailed report saved to: {report_filename}")

    # Summary status
    if len(wrong_frame_files) == 0 and len(error_files) == 0:
        print("\n🎉 ALL CLIPS ARE PERFECT! 450 frames each, properly encoded!")
    elif len(wrong_frame_files) == 0:
        print(f"\n✅ All clips have correct 450 frames, but {len(error_files)} had errors.")
    else:
        print(f"\n⚠️ {len(wrong_frame_files)} clips need attention (wrong frame count).")

if __name__ == "__main__":
    main()
