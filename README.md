# AVA Dataset Utilities

This repository contains utilities for working with the [AVA](https://research.google.com/ava/) video dataset, including:

- **ava_downloader.py**: Download AVA videos from the official source.
- **ava_sampler.py**: Sample clips from downloaded AVA videos based on annotation timestamps.
- **generate_csv.py**: Generate a CSV listing valid sampled clips for downstream tasks.
- **annotations/**: Official AVA annotation files (no video data included).

## Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd ava-dataset-utils
   ```

2. **Install dependencies:**
   - Python 3.7+
   - [ffmpeg](https://ffmpeg.org/) and [ffprobe](https://ffmpeg.org/ffprobe.html) (for video processing)
   - Python packages:
     ```sh
     pip install requests tqdm av
     ```

3. **Directory Structure:**
   - Place annotation files in the `annotations/` folder.
   - Downloaded videos will be stored in `videos/` (created by the scripts).
   - Sampled clips will be stored in `sampled_clips/` (created by the scripts).

## Usage

### 1. Download AVA Videos

Edit `ava_downloader.py` to set the correct split (`trainval` or `test`) and run:

```sh
python ava_downloader.py
```

Videos will be downloaded to `ava/videos/<split>/`.

### 2. Sample Clips

Edit `ava/ava_sampler.py` to set paths if needed, then run:

```sh
python ava/ava_sampler.py
```

Sampled clips will be saved in `ava/sampled_clips/`.

### 3. Generate CSV of Valid Clips

```sh
python generate_csv.py
```

This will create `ava/ava_clips_5s_list.csv` listing all valid clips.

## Notes

- **Videos are not included** in this repository. Download them using the provided scripts.
- **Kinetics dataset is not handled** here.
- Make sure you have enough disk space for videos and clips.

---

**Contributions welcome!**