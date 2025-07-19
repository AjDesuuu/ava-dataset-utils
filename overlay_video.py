import os
import pickle
import cv2
import random
from tqdm import tqdm

# === CONFIGURATION ===
PKL_DIR = "ava/flagged_masks"  # or ava/semantic_masks
MP4_DIR = "ava/sampled_clips"
OUTPUT_DIR = "ava/sample_overlays"
NUM_VIDEOS_TO_EXPORT = 10  # adjustable limit

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCTION DEFINITIONS ===

def draw_boxes(frame, detections):
    for det in detections:
        bbox = det['bbox']
        conf = det.get('conf', 1.0)
        class_id = det.get('class_id', -1)

        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'ID:{class_id} {conf:.2f}'
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def process_video(pkl_path, video_path, output_path):
    with open(pkl_path, 'rb') as f:
        mask_data = pickle.load(f)

    frames_detections = mask_data.get('frames', {})

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frames_detections.get(frame_idx, [])
        frame = draw_boxes(frame, detections)
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()


# === MAIN PROCESSING ===

pkl_files = [f for f in os.listdir(PKL_DIR) if f.endswith('_mask.pkl')]
random.shuffle(pkl_files)  # randomize selection

exported_count = 0

for pkl_filename in tqdm(pkl_files, desc="Processing videos"):
    if exported_count >= NUM_VIDEOS_TO_EXPORT:
        break

    base_filename = pkl_filename.replace('_mask.pkl', '')
    video_filename = f"{base_filename}.mp4"

    pkl_path = os.path.join(PKL_DIR, pkl_filename)
    video_path = os.path.join(MP4_DIR, video_filename)

    if not os.path.exists(video_path):
        print(f"Video not found for {base_filename}, skipping.")
        continue

    output_filename = f"{base_filename}_overlayed_mask.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    process_video(pkl_path, video_path, output_path)
    exported_count += 1

print(f"Exported {exported_count} overlayed videos to '{OUTPUT_DIR}'.")
