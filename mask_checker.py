import os
import pickle
import shutil
from tqdm import tqdm
import csv
from collections import Counter

# === CONFIGURATION ===
SEMANTIC_MASK_DIR = "ava/semantic_masks"
FLAGGED_DIR = "ava/flagged_masks"
REPORT_DIR = "Validation Report"

VALIDATION_CSV = os.path.join(REPORT_DIR, "validation_report.csv")
STATISTICS_TXT = os.path.join(REPORT_DIR, "dataset_statistics.txt")
SUDDEN_CHANGE_CSV = os.path.join(REPORT_DIR, "sudden_change_report.csv")
ZERO_DETECTION_TXT = os.path.join(REPORT_DIR, "zero_detection_pkl.txt")
SUDDEN_CHANGE_TXT = os.path.join(REPORT_DIR, "sudden_change_pkl.txt")

REQUIRED_KEYS = {"clip_id", "frames"}
SUDDEN_CHANGE_THRESHOLD = 0.5


def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


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
    errors = []
    object_class_counter = Counter()
    sudden_changes = []
    total_frames_in_file = 0
    total_no_object_frames = 0

    if not isinstance(data, dict):
        return False, ["Data is not a dictionary"], object_class_counter, sudden_changes, 0, 0

    allowed_keys = REQUIRED_KEYS
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        errors.append(f"Unexpected keys in data: {extra_keys}")

    if not REQUIRED_KEYS.issubset(data.keys()):
        errors.append(f"Missing required keys: {REQUIRED_KEYS - set(data.keys())}")

    if not isinstance(data.get("clip_id"), str):
        errors.append("clip_id is not a string")

    if not isinstance(data.get("frames"), dict):
        errors.append("frames is not a dictionary")

    if isinstance(data.get("frames"), dict) and len(data["frames"]) == 0:
        errors.append("frames dictionary is empty")

    if errors:
        return False, errors, object_class_counter, sudden_changes, 0, 0

    previous_count = None
    total_frames_in_file = len(data["frames"])

    for frame_idx, detections in data["frames"].items():
        detection_count = len(detections)

        if previous_count is not None:
            change_ratio = abs(detection_count - previous_count) / max(previous_count, 1)
            if change_ratio >= SUDDEN_CHANGE_THRESHOLD:
                sudden_changes.append((frame_idx, f"Sudden change from {previous_count} to {detection_count} detections"))
        previous_count = detection_count

        if detection_count == 0:
            errors.append((frame_idx, "has zero detections"))
            total_no_object_frames += 1
            continue

        for det in detections:
            if not validate_detection(det):
                errors.append((frame_idx, "Invalid detection format"))
                continue
            class_id = det["class_id"]
            object_class_counter[class_id] += 1

    if errors:
        return False, errors, object_class_counter, sudden_changes, total_frames_in_file, total_no_object_frames

    return True, ["Valid"], object_class_counter, sudden_changes, total_frames_in_file, total_no_object_frames


def load_pickle_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Unpickling error: {e}"


def copy_flagged_files(flagged_files, src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname, _ in flagged_files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dest_dir, fname)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Failed to copy {fname}: {e}")


def write_csv_report(flagged_files, output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Frame Number", "Error Message"])
        for fname, reasons in flagged_files:
            for reason in reasons:
                if isinstance(reason, tuple) and len(reason) == 2:
                    frame_number, error_msg = reason
                else:
                    frame_number, error_msg = "N/A", str(reason)
                writer.writerow([fname, frame_number, error_msg])


def write_sudden_change_csv(sudden_changes, output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Frame Number", "Description"])
        for fname, changes in sudden_changes.items():
            for frame_idx, description in changes:
                writer.writerow([fname, frame_idx, description])


def write_list_file(file_list, output_path):
    with open(output_path, mode="w", encoding="utf-8") as f:
        for fname in sorted(file_list):
            f.write(f"{fname}\n")


def write_dataset_statistics(stats, output_path):
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("📊 Dataset Statistics:\n")
        f.write(f"Total PKL files processed: {stats['total_pkl_files']}\n")
        f.write(f"Total frames processed: {stats['total_frames']}\n")
        f.write(f"Total frames with no detections: {stats['total_no_object_frames']}\n")

        if stats['total_frames'] > 0:
            empty_frame_percent = (stats['total_no_object_frames'] / stats['total_frames']) * 100
            f.write(f"Percentage of frames with no detections: {empty_frame_percent:.2f}%\n")

        f.write("\nObject Class Occurrences (class_id : count):\n")
        for class_id, count in stats['object_class_counter'].most_common():
            f.write(f"Class {class_id}: {count}\n")


def validate_all_masks():
    flagged_files = []
    sudden_changes_all = {}
    object_class_counter = Counter()

    total_frames = 0
    total_no_object_frames = 0

    print(f"\n📁 Scanning: {SEMANTIC_MASK_DIR} ...")

    pkl_files = [f for f in os.listdir(SEMANTIC_MASK_DIR) if f.endswith(".pkl")]

    for file_name in tqdm(pkl_files):
        file_path = os.path.join(SEMANTIC_MASK_DIR, file_name)
        data, load_error = load_pickle_file(file_path)

        if load_error:
            flagged_files.append((file_name, [load_error]))
            continue

        is_valid, messages, file_class_counter, sudden_changes, frames_in_file, no_obj_frames = validate_mask_file(data)

        if sudden_changes:
            sudden_changes_all[file_name] = sudden_changes

        object_class_counter.update(file_class_counter)
        total_frames += frames_in_file
        total_no_object_frames += no_obj_frames

        if not is_valid:
            flagged_files.append((file_name, messages))

    stats = {
        "total_pkl_files": len(pkl_files),
        "total_frames": total_frames,
        "total_no_object_frames": total_no_object_frames,
        "object_class_counter": object_class_counter
    }

    return flagged_files, sudden_changes_all, stats


def extract_zero_detection_files(validation_csv):
    files = set()
    if os.path.exists(validation_csv):
        with open(validation_csv, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "has zero detections" in row["Error Message"]:
                    files.add(row["File Name"])
    return files


def extract_sudden_change_files(sudden_change_csv):
    files = set()
    if os.path.exists(sudden_change_csv):
        with open(sudden_change_csv, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                files.add(row["File Name"])
    return files


def main():
    ensure_report_dir()

    flagged_files, sudden_changes_all, stats = validate_all_masks()

    print("\nGenerating reports...")

    copy_flagged_files(flagged_files, SEMANTIC_MASK_DIR, FLAGGED_DIR)
    write_csv_report(flagged_files, VALIDATION_CSV)
    write_sudden_change_csv(sudden_changes_all, SUDDEN_CHANGE_CSV)
    write_dataset_statistics(stats, STATISTICS_TXT)

    zero_detection_files = extract_zero_detection_files(VALIDATION_CSV)
    sudden_change_files = extract_sudden_change_files(SUDDEN_CHANGE_CSV)

    write_list_file(zero_detection_files, ZERO_DETECTION_TXT)
    write_list_file(sudden_change_files, SUDDEN_CHANGE_TXT)

    print("\n✅ All reports generated:")
    print(f" - {VALIDATION_CSV}")
    print(f" - {SUDDEN_CHANGE_CSV}")
    print(f" - {STATISTICS_TXT}")
    print(f" - {ZERO_DETECTION_TXT}")
    print(f" - {SUDDEN_CHANGE_TXT}")


if __name__ == "__main__":
    main()
