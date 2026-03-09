# Finetuning PixelPlayer (Sound of Pixels) on AVSpeech for Human Speech Separation

## Overview

This report details how to finetune the pretrained PixelPlayer model — originally trained on the MUSIC dataset for musical instrument separation — on the **AVSpeech dataset**, enabling it to perform pixel-level audio-visual source separation for **human speakers**. The report covers the architectural differences, dataset preparation, required code changes, training strategy, and evaluation approach.

---

## 1. Understanding the Pretrained Model

### 1.1 Architecture Summary

PixelPlayer consists of three sub-networks:

| Component | Architecture | Role |
|---|---|---|
| **Video Analysis Network** (`net_frame`) | Dilated ResNet-18 (ImageNet pretrained) | Extracts per-pixel visual features from video frames |
| **Audio Analysis Network** (`net_sound`) | U-Net (7 down + 7 up convolutions with skip connections) | Decomposes mixed audio spectrogram into K component feature maps |
| **Audio Synthesizer** (`net_synthesizer`) | Linear layer (K weights + 1 bias) | Fuses visual and audio features into a per-pixel spectrogram mask |

### 1.2 Training Paradigm: Mix-and-Separate

The model is trained with a **self-supervised Mix-and-Separate framework**:
1. N videos are sampled randomly from the training set
2. Their audio tracks are summed: `S_mix = S_1 + S_2 + ... + S_N`
3. The model learns to separate each source `S_n` conditioned on the corresponding visual frames `I_n`
4. No manual labels are used — supervision comes entirely from the known mixture structure

### 1.3 Key Hyperparameters (Default MUSIC Training)

```
--arch_frame resnet18dilated
--num_channels 32
--num_frames 3
--stride_frames 24
--frameRate 8
--audLen 65535
--audRate 11025
--binary_mask 1
--loss bce
--weighted_loss 1
--num_mix 2
--log_freq 1
--batch_size_per_gpu 20
--lr_frame 1e-4
--lr_sound 1e-3
--lr_synthesizer 1e-3
--num_epoch 100
--lr_steps 40 80
```

### 1.4 Data Format Expected by the Codebase

The model expects an index CSV file (`train.csv` / `val.csv`) with one row per video in the format:

```
AUDIO_PATH,FRAMES_PATH,NUMBER_OF_FRAMES
./data/audio/acoustic_guitar/videoID.mp3,./data/frames/acoustic_guitar/videoID.mp4,1580
```

Frames are pre-extracted JPEGs at **8 fps**, stored in per-video subdirectories. Audio is stored as `.mp3` or `.wav` files at **11,025 Hz**.

---

## 2. Understanding the AVSpeech Dataset

### 2.1 Dataset Description

AVSpeech is a large-scale audio-visual dataset from Google containing roughly **4,700 hours** of clean speech video clips, drawn from ~290,000 YouTube videos. Key characteristics:

- Each clip is **3–10 seconds** long
- Each clip contains a **single visible speaking person** with no background noise
- The dataset spans ~150,000 distinct speakers across many languages, poses, and lighting conditions
- The train and test sets have **disjoint speakers** (speaker-independent)

### 2.2 CSV Format

The dataset is distributed as two CSV files (train and test). Each row has the format:

```
YouTube_ID, start_time, end_time, face_x, face_y
```

- `face_x` and `face_y`: normalized (0.0–1.0) coordinates of the speaker's face center at the start of the segment
- These coordinates are critical — they tell you **where the speaking face is** in each video

### 2.3 HuggingFace Version (`ProgramComputer/avspeech-visual-audio`)

The HuggingFace version of AVSpeech preprocesses and hosts the audio/video directly, potentially providing pre-extracted audio and frames, making it more accessible than raw YouTube downloading. It follows the same conceptual schema as the original AVSpeech CSV data.

### 2.4 Why AVSpeech is a Good Fit

The Mix-and-Separate framework maps almost perfectly onto AVSpeech:
- Each clip has exactly **one clean speech source** (a single speaker)
- Multiple clips can be mixed synthetically to create cocktail-party scenarios
- The face location metadata provides a natural **visual anchor** for the audio source

---

## 3. The Domain Gap: Instruments vs. People

### 3.1 Visual Domain Differences

| Aspect | MUSIC (Instruments) | AVSpeech (People) |
|---|---|---|
| Visual appearance | Static instruments (guitar, violin, etc.) | Dynamic human faces with lip movement |
| Sound source region | Instrument body / strings / bell | Mouth / face region |
| Visual cues | Shape, texture of instrument | Lip movement, face pose, identity |
| Frame diversity | Moderate (mostly performance settings) | High (any setting, any lighting) |

The pretrained ResNet-18 video network has learned to activate on **instrument shapes and textures**. For speech, the model needs to attend to **faces**, especially lip regions — a significantly different visual signal. The video network will need the most retraining.

### 3.2 Audio Domain Differences

| Aspect | MUSIC | AVSpeech |
|---|---|---|
| Frequency range | 0–5.5 kHz (instrument harmonics, log-scaled) | 80 Hz–8 kHz (speech formants) |
| Temporal structure | Sustained notes, rhythmic patterns | Short phonemic bursts, pauses |
| Spectral structure | Harmonic stacks, clear pitch | Formant structure (F1/F2/F3) |

The audio U-Net has learned instrument spectral patterns. While the general masking approach transfers, the learned filter banks will adapt to speech spectral structure during finetuning.

### 3.3 Sample Rate Consideration

The original model uses **11,025 Hz** (preserving up to 5.5 kHz). For speech, this is acceptable — the most intelligibility-critical frequencies (300 Hz–3.4 kHz) are well within range. However, **16,000 Hz** is the standard for speech processing and would capture more speech-relevant high-frequency content. This is a tunable decision.

---

## 4. Data Preparation Pipeline

### 4.1 Download AVSpeech Data

**Option A: From HuggingFace (Recommended for Colab)**

```python
from datasets import load_dataset
dataset = load_dataset("ProgramComputer/avspeech-visual-audio", split="train")
```

**Option B: Download raw videos from YouTube using the official CSV**

```bash
# Install dependencies
pip install yt-dlp

# Download using the AVSpeech CSV
python download_avspeech.py \
    --csv avspeech_train.csv \
    --output_dir ./data/avspeech/ \
    --num_workers 8
```

A download script (`download_avspeech.py`) should:
1. Read each row: `(yt_id, t_start, t_end, face_x, face_y)`
2. Download only the clip segment using `yt-dlp` with `--download-sections`
3. Save audio as `.wav` at 11,025 Hz (or 16,000 Hz if changing sample rate)
4. Save video frames as JPEGs

### 4.2 Preprocess Videos into PixelPlayer Format

Extract frames at 8 fps and audio at 11,025 Hz:

```bash
# For each video clip:
ffmpeg -i clip.mp4 -vf fps=8 frames/%06d.jpg
ffmpeg -i clip.mp4 -ar 11025 -ac 1 audio.wav
```

### 4.3 Directory Structure

Organize data to match PixelPlayer's expected structure. Since AVSpeech is speaker-centric rather than category-centric, use a flat or speaker-id-based structure:

```
data/
├── audio/
│   └── speaker/
│       ├── <yt_id>_<start>_<end>.wav
│       ├── ...
└── frames/
    └── speaker/
        ├── <yt_id>_<start>_<end>/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        └── ...
```

Since PixelPlayer doesn't use category labels during training (purely self-supervised), all clips can be placed under a single `speaker` folder — the category subdivision is only used for evaluation.

### 4.4 Create Index CSV Files

```python
# create_avspeech_index.py
import os, glob, csv

audio_dir = "./data/audio/speaker"
frames_dir = "./data/frames/speaker"
rows = []

for audio_file in glob.glob(os.path.join(audio_dir, "*.wav")):
    clip_id = os.path.splitext(os.path.basename(audio_file))[0]
    frames_path = os.path.join(frames_dir, clip_id)
    if os.path.isdir(frames_path):
        n_frames = len(glob.glob(os.path.join(frames_path, "*.jpg")))
        if n_frames > 0:
            rows.append([audio_file, frames_path, n_frames])

# Shuffle and split
import random
random.shuffle(rows)
n_train = int(0.9 * len(rows))

with open("train.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(rows[:n_train])

with open("val.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(rows[n_train:])
```

### 4.5 Face Crop Consideration (Important)

The face location metadata `(face_x, face_y)` from AVSpeech is valuable but **not directly used** by the base PixelPlayer architecture, which uses entire video frames. You have two options:

**Option A (Simpler — Recommended First):** Use full frames as-is. The model will learn to attend to faces naturally through the Mix-and-Separate objective.

**Option B (Better performance):** Pre-crop frames to a fixed region around the face using `(face_x, face_y)` metadata, then pass the cropped face thumbnail to the video network. This more directly mirrors how "Looking to Listen" (Google's concurrent work) operates and reduces irrelevant visual noise.

---

## 5. Code Changes Required

### 5.1 Loading Pretrained Weights (No Code Change Needed)

PixelPlayer's `arguments.py` already supports weight loading for finetuning via:

```bash
--weights_sound  path/to/pretrained_net_sound.pth
--weights_frame  path/to/pretrained_net_frame.pth
--weights_synthesizer  path/to/pretrained_net_synthesizer.pth
```

Download the pretrained weights:
```bash
./scripts/download_trained_model.sh
```

This will provide three weight files: `net_sound.pth`, `net_frame.pth`, and `net_synthesizer.pth`.

### 5.2 Dataset Loader (`dataset/music.py` or equivalent)

The existing dataset loader reads from `train.csv` / `val.csv` and extracts audio + frames. **No changes are needed** as long as the AVSpeech data is formatted identically. Verify that:

1. Audio files are readable by `librosa` or `scipy` at 11,025 Hz
2. Frame files are accessible as numbered JPEGs

### 5.3 Optional: Face-Crop Preprocessing in the Dataset Loader

If using **Option B** (face crops), modify the dataset loader's frame loading logic to:

```python
# In the frame loading section of dataset/music.py:
# Load face metadata from a sidecar file or CSV
face_x, face_y = load_face_coords(video_id)

# Crop around face center
crop_size = 112  # pixels
cx = int(face_x * frame_width)
cy = int(face_y * frame_height)
frame = frame[
    max(0, cy - crop_size//2):min(frame_height, cy + crop_size//2),
    max(0, cx - crop_size//2):min(frame_width, cx + crop_size//2)
]
frame = cv2.resize(frame, (224, 224))
```

### 5.4 Optional: Audio Sample Rate Change

If switching to 16,000 Hz for better speech quality:

In `arguments.py` / training script, change:
```bash
--audRate 16000
--audLen 96000   # ~6 seconds at 16kHz
```

In `main.py`, the STFT parameters may need adjustment:
```python
# Original (11025 Hz):
stft_frame = 1022
stft_hop = 256

# Adjusted (16000 Hz, similar time resolution):
stft_frame = 1024
stft_hop = 256
```

### 5.5 Freezing Strategy: What to Freeze vs. Finetune

The key strategic decision is which layers to freeze during finetuning:

**Recommended Strategy: Freeze audio U-Net, finetune video network**

```
net_frame (ResNet-18):      FINETUNE — domain gap is largest here (instruments → faces)
net_sound (U-Net):          FREEZE first epoch, then unfreeze — audio masking logic transfers
net_synthesizer (Linear):   ALWAYS FINETUNE — small, fast to adapt
```

This can be controlled via different learning rates:

```bash
--lr_frame 1e-4        # Higher LR for frame network (needs most adaptation)
--lr_sound 1e-5        # Lower LR or zero to freeze sound network initially
--lr_synthesizer 1e-3  # Full LR for synthesizer
```

To fully freeze `net_sound` for the first phase, add to `main.py`:

```python
# Freeze net_sound for first N epochs
if epoch < args.freeze_sound_epochs:
    for param in net_sound.parameters():
        param.requires_grad = False
else:
    for param in net_sound.parameters():
        param.requires_grad = True
```

---

## 6. Training Configuration

### 6.1 Recommended Finetuning Script (`scripts/finetune_AVSpeech.sh`)

```bash
#!/bin/bash

# Paths
OPTS=""
OPTS+="--id AVSpeech-finetune "
OPTS+="--list_train ./data/avspeech/train.csv "
OPTS+="--list_val ./data/avspeech/val.csv "

# Load pretrained weights
OPTS+="--weights_sound ./ckpt/MUSIC_pretrained/net_sound_best.pth "
OPTS+="--weights_frame ./ckpt/MUSIC_pretrained/net_frame_best.pth "
OPTS+="--weights_synthesizer ./ckpt/MUSIC_pretrained/net_synthesizer_best.pth "

# Architecture (must match pretrained)
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "

# Loss (binary mask with BCE)
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "

# Audio/visual settings
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# Learning rates (lower than scratch training)
OPTS+="--lr_frame 5e-5 "
OPTS+="--lr_sound 1e-4 "
OPTS+="--lr_synthesizer 5e-4 "
OPTS+="--num_epoch 50 "
OPTS+="--lr_steps 20 40 "

# Resources
OPTS+="--num_gpus 1 "
OPTS+="--workers 8 "
OPTS+="--batch_size_per_gpu 16 "

# Logging
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 20 "
OPTS+="--num_val 256 "

python -u main.py $OPTS
```

### 6.2 Two-Phase Training Strategy

For best results, use a two-phase approach:

**Phase 1 (Epochs 1–15): Adapt the visual network only**
- Freeze `net_sound` (set `lr_sound 0`)
- Train `net_frame` and `net_synthesizer`
- Goal: Teach the visual encoder to respond to faces/mouths instead of instruments

**Phase 2 (Epochs 16–50): Joint finetuning**
- Unfreeze all networks with low learning rates
- Goal: Co-adapt all networks for speech separation

### 6.3 Data Scale Recommendations

AVSpeech has ~290,000 videos. For Colab/limited compute, a practical subset:

| Scale | Videos | Expected GPU Hours (1x T4) |
|---|---|---|
| Small (proof of concept) | 5,000 | ~8 hours |
| Medium (good results) | 30,000 | ~48 hours |
| Full (best results) | 290,000 | ~400+ hours |

For Colab, start with **5,000–10,000 clips** to validate the pipeline, then scale up if using Colab Pro+ or a cloud VM.

---

## 7. Evaluation

### 7.1 Quantitative Metrics

The same metrics used in the original paper apply:
- **NSDR** (Normalized Signal-to-Distortion Ratio): improvement over the mixture baseline
- **SIR** (Signal-to-Interference Ratio): how well other speakers are suppressed
- **SAR** (Signal-to-Artifact Ratio): absence of separation artifacts

Run evaluation using the existing script pattern:

```bash
python -u main.py \
    --mode eval \
    --weights_sound ./ckpt/AVSpeech-finetune/net_sound_best.pth \
    --weights_frame ./ckpt/AVSpeech-finetune/net_frame_best.pth \
    --weights_synthesizer ./ckpt/AVSpeech-finetune/net_synthesizer_best.pth \
    --list_val ./data/avspeech/val.csv \
    --num_mix 2 \
    --audRate 11025 \
    --audLen 65535
```

### 7.2 Qualitative Evaluation

Use the existing HTML visualization output (`ckpt/MODEL_ID/visualization/`) to inspect:
- Heatmaps of sound energy per pixel (should highlight the face/mouth region)
- Waveform comparisons of mixed vs. separated audio
- Spectrogram masks (should show speech formant patterns)

### 7.3 Expected Performance Baseline

For reference, "Looking to Listen" (Ephrat et al. 2018), which uses a dedicated face-embedding approach on AVSpeech, achieves strong perceptual separation. PixelPlayer finetuned on AVSpeech will likely produce competitive results for the pixel-level localization task, though dedicated face-network approaches may outperform it for pure separation quality.

---

## 8. Potential Issues and Solutions

### 8.1 Lip Sync / Temporal Misalignment

**Issue:** The model uses only 3 frames sampled with stride 24 (at 8 fps = 3 seconds apart). For speech, lip movements are fast — this sparse sampling may miss important visual cues.

**Solution:** Increase `--num_frames` and reduce `--stride_frames`:
```bash
--num_frames 5
--stride_frames 8    # 1-second stride at 8fps
```

### 8.2 Silent Background Regularization

The original training uses ADE20K silent background images to prevent the model from hallucinating sounds from non-speaking regions. For AVSpeech finetuning, use the same technique:
```bash
# Already handled in main.py via --ade20k_dir argument
--ade20k_dir ./data/ade20k/
```

### 8.3 Speaker Identity Collapse

**Issue:** With many diverse speakers, the visual network may learn a generic "face" detector rather than a speaker-specific one, degrading separation when faces look similar.

**Solution:** The Mix-and-Separate framework naturally handles this — since the model must separate two different voices conditioned on two different faces in each training step, it is forced to learn discriminative face features.

### 8.4 PyTorch Version Compatibility

The original code requires `PyTorch >= 0.4.0`. Modern PyTorch (2.x) is fully backward compatible with this codebase. No changes are needed for PyTorch 2.x, though you should verify that `warpgrid` and custom grid sampling operations still work as expected.

### 8.5 Incomplete AVSpeech Downloads

Many YouTube videos from the original AVSpeech dataset have been deleted or made private since 2018. The HuggingFace version (`ProgramComputer/avspeech-visual-audio`) mitigates this by hosting the processed data directly.

---

## 9. Step-by-Step Quickstart Summary

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hangzhaomit/Sound-of-Pixels
   cd Sound-of-Pixels
   pip install -r requirements.txt
   ```

2. **Download pretrained MUSIC weights:**
   ```bash
   ./scripts/download_trained_model.sh
   ```

3. **Prepare AVSpeech data** (using HuggingFace or raw download), organize into the expected directory structure, and generate `train.csv` / `val.csv`.

4. **Run Phase 1 finetuning** (visual network only, ~15 epochs).

5. **Run Phase 2 finetuning** (all networks jointly, ~35 more epochs).

6. **Evaluate** using `--mode eval` and inspect HTML visualizations.

7. **Test on custom video** using the PixelPlayer inference scripts.

---

## 10. References

- Zhao et al., *The Sound of Pixels*, ECCV 2018 — [[paper]](https://arxiv.org/abs/1804.03160) [[code]](https://github.com/hangzhaomit/Sound-of-Pixels)
- Ephrat et al., *Looking to Listen at the Cocktail Party*, SIGGRAPH 2018 — [[paper]](https://arxiv.org/abs/1804.03619) [[dataset]](https://looking-to-listen.github.io/avspeech/)
- AVSpeech HuggingFace dataset — [[link]](https://huggingface.co/datasets/ProgramComputer/avspeech-visual-audio)
