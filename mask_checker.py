import os
import pickle
import shutil
from tqdm import tqdm
import yaml

def load_paths(config_path="paths.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
paths = load_paths()

SEMANTIC_MASK_DIR = os.path.join(BASE_DIR, paths["semantic_masks_dir"])
FLAGGED_DIR = os.path.join(BASE_DIR, paths["flagged_masks_dir"])
# use savample overlays folder to create the video sample with the pkl overlay. 
SAMPLE_OVERLAYS_DIR = os.path.join(BASE_DIR, paths["sample_overlays_dir"]) 

REQUIRED_KEYS = {"clip_id", "frames"}

def validate_detection(det):
    return (
        isinstance(det, dict)
        and "bbox" in det
        and "class_id" in det
        and "conf" in det
        and isinstance(det["bbox"], list)
        and len(det["bbox"]) == 4
        and all(isinstance(v, (int, float)) for v in det["bbox"])
        and isinstance(det["class_id"], int)
        and isinstance(det["conf"], float)
    )

def validate_mask_file(data):
    if not isinstance(data, dict):
        return False, "Data is not a dictionary"

    # Unexpected keys check
    allowed_keys = REQUIRED_KEYS
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        return False, f"Unexpected keys in data: {extra_keys}"

    if not REQUIRED_KEYS.issubset(data.keys()):
        return False, f"Missing required keys: {REQUIRED_KEYS - set(data.keys())}"

    if not isinstance(data["clip_id"], str):
        return False, "clip_id is not a string"

    if not isinstance(data["frames"], dict):
        return False, "frames is not a dictionary"

    if len(data["frames"]) == 0:
        return False, "frames dictionary is empty"

    # Sequential frames check
    frame_indices = list(data["frames"].keys())
    if sorted(frame_indices) != list(range(len(frame_indices))):
        return False, f"Frame indices are not sequential starting from 0: {frame_indices}"

    for frame_idx, detections in data["frames"].items():
        if not isinstance(frame_idx, int):
            return False, f"Frame index '{frame_idx}' is not an int"
        if not isinstance(detections, list):
            return False, f"Detections for frame {frame_idx} is not a list"
        if len(detections) == 0:
            return False, f"Frame {frame_idx} has zero detections"
        if len(detections) > 100:
            return False, f"Frame {frame_idx} has suspiciously many detections: {len(detections)}"
        # Duplicate detection check
        seen_dets = set()
        for det in detections:
            if not validate_detection(det):
                return False, f"Invalid detection format in frame {frame_idx}"
            # Bbox/class/conf validity
            bbox = det["bbox"]
            class_id = det["class_id"]
            conf = det["conf"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return False, f"Invalid bbox format in frame {frame_idx}: {bbox}"
            x1, y1, x2, y2 = bbox
            if not (x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0):
                return False, f"Invalid bbox coordinates in frame {frame_idx}: {bbox}"
            if not (isinstance(class_id, int) and class_id >= 0):
                return False, f"Invalid class_id in frame {frame_idx}: {class_id}"
            if not (isinstance(conf, float) and 0.0 <= conf <= 1.0):
                return False, f"Invalid confidence in frame {frame_idx}: {conf}"
            # Duplicate detection check (tuple of bbox, class_id, conf)
            det_tuple = (tuple(bbox), class_id, conf)
            if det_tuple in seen_dets:
                return False, f"Duplicate detection in frame {frame_idx}: {det_tuple}"
            seen_dets.add(det_tuple)
    return True, "Valid"

def main():
    clip_ids = set()
    duplicate_ids = set()
    flagged_files = []

    os.makedirs(FLAGGED_DIR, exist_ok=True)

    print(f"📁 Scanning: {SEMANTIC_MASK_DIR}\n")

    for file_name in tqdm(os.listdir(SEMANTIC_MASK_DIR)):
        if not file_name.endswith(".pkl"):
            continue

        file_path = os.path.join(SEMANTIC_MASK_DIR, file_name)

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            flagged_files.append((file_name, f"Unpickling error: {e}"))
            continue

        is_valid, message = validate_mask_file(data)
        if not is_valid:
            flagged_files.append((file_name, message))
            continue

        clip_id = data["clip_id"]
        if clip_id in clip_ids:
            duplicate_ids.add(clip_id)
            flagged_files.append((file_name, f"Duplicate clip_id: {clip_id}"))
        else:
            clip_ids.add(clip_id)

    print("\n📋 Validation Report:")

    if not flagged_files:
        print("✅ All .pkl files are valid.")
    else:
        for fname, reason in flagged_files:
            print(f"❌ {fname}: {reason}")

        print(f"\n⚠️ Total flagged files: {len(flagged_files)}")
        if duplicate_ids:
            print(f"⚠️ Duplicate clip IDs found: {duplicate_ids}")

    # Optional: copy flagged files to separate directory
    for fname, _ in flagged_files:
        src = os.path.join(SEMANTIC_MASK_DIR, fname)
        dst = os.path.join(FLAGGED_DIR, fname)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Failed to copy {fname}: {e}")

if __name__ == "__main__":
    main()
