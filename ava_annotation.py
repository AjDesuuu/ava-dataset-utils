#!/usr/bin/env python3
"""
AVA Annotation Downloader
Downloads and organizes AVA dataset annotations properly.
Based on Facebook's download_annotations.sh but with Python implementation.
"""

import os
import urllib.request
import yaml
from pathlib import Path
import sys

def load_paths(config_path="paths.yaml"):
    """Load paths from configuration file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        print(f"📥 Downloading: {os.path.basename(destination)}")
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded: {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(destination)}: {e}")
        return False

def main():
    print("🎯 AVA Annotation Downloader")
    
    # Load configuration
    paths = load_paths()
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_dir = os.path.join(base_dir, paths["annotations_dir"])
    
    # Create annotations directory
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"📁 Annotations directory: {annotations_dir}")
    
    # AVA annotation files to download
    base_url = "https://research.google.com/ava/download"
    predicted_boxes_url = "https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations"
    
    files_to_download = [
        # Core annotation files
        ("ava_train_v2.2.csv", f"{base_url}/ava_train_v2.2.csv"),
        ("ava_val_v2.2.csv", f"{base_url}/ava_val_v2.2.csv"),
        ("ava_test_v2.2.csv", f"{base_url}/ava_test_v2.2.csv"),
        
        # Action list and metadata
        ("ava_action_list_v2.2.pbtxt", f"{base_url}/ava_action_list_v2.2.pbtxt"),
        ("ava_action_list_v2.2_for_activitynet_2019.pbtxt", 
         f"{base_url}/ava_action_list_v2.2_for_activitynet_2019.pbtxt"),
        
        # Excluded timestamps
        ("ava_train_excluded_timestamps_v2.2.csv", 
         f"{base_url}/ava_train_excluded_timestamps_v2.2.csv"),
        ("ava_val_excluded_timestamps_v2.2.csv", 
         f"{base_url}/ava_val_excluded_timestamps_v2.2.csv"),
        ("ava_test_excluded_timestamps_v2.2.csv", 
         f"{base_url}/ava_test_excluded_timestamps_v2.2.csv"),
        
        # Included timestamps
        ("ava_included_timestamps_v2.2.txt", 
         f"{base_url}/ava_included_timestamps_v2.2.txt"),
        
        # Predicted bounding boxes from Facebook's repository
        ("ava_train_predicted_boxes.csv", 
         f"{predicted_boxes_url}/ava_train_predicted_boxes.csv"),
        ("ava_val_predicted_boxes.csv", 
         f"{predicted_boxes_url}/ava_val_predicted_boxes.csv"),
        ("ava_test_predicted_boxes.csv", 
         f"{predicted_boxes_url}/ava_test_predicted_boxes.csv"),
    ]
    
    # Download files
    success_count = 0
    total_files = len(files_to_download)
    
    for filename, url in files_to_download:
        destination = os.path.join(annotations_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(destination):
            print(f"✔️ Already exists: {filename}")
            success_count += 1
            continue
        
        if download_file(url, destination):
            success_count += 1
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{total_files}")
    
    if success_count == total_files:
        print("🎉 All annotation files downloaded successfully!")
    else:
        failed_count = total_files - success_count
        print(f"⚠️ {failed_count} files failed to download. Please check the URLs.")
    
    # Verify essential files
    essential_files = ["ava_train_v2.2.csv", "ava_val_v2.2.csv", "ava_action_list_v2.2.pbtxt"]
    missing_essential = []
    
    for filename in essential_files:
        filepath = os.path.join(annotations_dir, filename)
        if not os.path.exists(filepath):
            missing_essential.append(filename)
    
    if missing_essential:
        print(f"\n⚠️ Missing essential files: {missing_essential}")
        print("Please download these manually or check the URLs.")
    else:
        print("\n✅ All essential annotation files are present!")

if __name__ == "__main__":
    main()