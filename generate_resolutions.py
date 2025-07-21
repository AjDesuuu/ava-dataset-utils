import os
import csv
import cv2
import yaml
from tqdm import tqdm
import concurrent.futures

def load_paths(config_path="paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

# Use the sampled_clips directory from paths.yaml
sampled_clips_dir = os.path.join(BASE_DIR, "ava", "sampled_clips")
output_csv = os.path.join(BASE_DIR, "ava", "ava_resolutions.csv")

# Supported video file extension
clip_extension = ".mp4"

def get_video_resolution(video_path):
    """Get video resolution using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return width, height
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None

def process_video_file(video_path):
    """Process a single video file and return its resolution data."""
    width, height = get_video_resolution(video_path)
    if width is not None and height is not None:
        # Get just the filename without path for consistency
        filename = os.path.basename(video_path)
        return [filename, width, height]
    return None

def scan_and_generate_resolutions_csv(sampled_clips_dir, output_csv):
    """Scan the sampled clips directory and generate a CSV with video resolutions."""
    print(f"📁 Scanning directory: {sampled_clips_dir}")
    
    # Gather all video files
    video_files = []
    for root, _, files in os.walk(sampled_clips_dir):
        for file in files:
            if file.endswith(clip_extension):
                video_files.append(os.path.join(root, file))
    
    print(f"🎥 Found {len(video_files)} video files")
    
    # Process videos in parallel
    valid_resolutions = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_video_file, video_files),
            total=len(video_files),
            desc="Extracting resolutions",
            unit="video"
        ))
    
    # Filter out failed results
    for result in results:
        if result is not None:
            valid_resolutions.append(result)
    
    # Write the resolutions to CSV file
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "width", "height"])  # CSV header
        for resolution_data in valid_resolutions:
            writer.writerow(resolution_data)
    
    print(f"📊 CSV file generated: {output_csv}")
    print(f"✅ Total videos processed: {len(valid_resolutions)}")
    
    # Print some statistics
    if valid_resolutions:
        widths = [r[1] for r in valid_resolutions]
        heights = [r[2] for r in valid_resolutions]
        unique_resolutions = set((r[1], r[2]) for r in valid_resolutions)
        
        print(f"📈 Resolution statistics:")
        print(f"   Unique resolutions: {len(unique_resolutions)}")
        print(f"   Width range: {min(widths)} - {max(widths)}")
        print(f"   Height range: {min(heights)} - {max(heights)}")
        print(f"   Most common resolutions:")
        
        # Count resolution frequencies
        resolution_counts = {}
        for w, h in unique_resolutions:
            count = sum(1 for r in valid_resolutions if r[1] == w and r[2] == h)
            resolution_counts[(w, h)] = count
        
        # Show top 5 most common resolutions
        sorted_resolutions = sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)
        for (w, h), count in sorted_resolutions[:5]:
            percentage = (count / len(valid_resolutions)) * 100
            print(f"     {w}x{h}: {count} videos ({percentage:.1f}%)")

if __name__ == "__main__":
    # Check if sampled_clips directory exists
    if not os.path.exists(sampled_clips_dir):
        print(f"❌ Error: Directory not found: {sampled_clips_dir}")
        print("Please ensure the sampled_clips directory exists.")
        exit(1)
    
    scan_and_generate_resolutions_csv(sampled_clips_dir, output_csv)
