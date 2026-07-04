A deep learning system that analyzes videos and determines whether they have been digitally manipulated — covering deepfakes, video splicing, and object insertion/removal. Built as an extension of an image forgery detection pipeline, adding temporal (time-based) reasoning on top of spatial (frame-level) analysis.


📖 Table of Contents


What Is Video Forgery?
How Detection Works — The Big Picture
Core Concepts Explained

Error Level Analysis (ELA)
EfficientNet-B4 (Spatial Encoder)
Bidirectional GRU (Temporal Head)
Optical Flow
Transfer Learning
Focal Loss



System Architecture
Project Structure
Setup
Dataset
Training
Inference
Ethical Use



What Is Video Forgery?

A forged video is one where content has been digitally altered after recording. There are three main categories this system targets:

1. Deepfakes (Face Swaps)

A person's face is replaced with another's using AI — typically a GAN (Generative Adversarial Network). The resulting video looks real but the face belongs to someone else. These are the hardest to detect because the entire face region is synthetically generated.


Example: A politician's face swapped onto someone else's body saying things they never said.



2. Video Splicing

Two or more video clips are cut and joined together. A scene from one video is inserted into another — often to change context, alter a timeline, or fabricate an event sequence.


Example: Cutting footage from two different protests together to make them look like one event.



3. Object Insertion / Removal

Objects, people, or elements are added into or removed from existing video frames — frame by frame — using compositing or inpainting techniques.


Example: Removing a watermark, adding a weapon to a scene, or inserting a person into footage.




How Detection Works

The system uses a two-stage pipeline: first analyze each frame individually (spatial), then look at how frames relate to each other over time (temporal).

Input Video
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: Frame-Level (Spatial)     │
│                                     │
│  Sample 64 frames  →  Face Crop     │
│         →  ELA Blend                │
│         →  EfficientNet-B4          │
│         →  1792-d feature vector    │
│             per frame               │
└──────────────────┬──────────────────┘
                   │ sequence of 64 feature vectors
                   ▼
┌─────────────────────────────────────┐
│  STAGE 2: Sequence-Level (Temporal) │
│                                     │
│  Bidirectional GRU → captures       │
│  cross-frame inconsistency          │
│         →  single verdict           │
└──────────────────┬──────────────────┘
                   │
                   ▼
            REAL / FAKE
           + Confidence %
           + Splice Report

The key insight: a forged video has inconsistencies that only become visible when you look across multiple frames, not just one. A deepfake face might look perfect in isolation, but its movement, lighting, and texture will subtly clash with the rest of the video across time.


Core Concepts Explained

1. Error Level Analysis (ELA)

What it is: A forensic technique that reveals regions in an image that have been digitally modified.

How it works:

Every image saved as JPEG undergoes lossy compression — it throws away some information to reduce file size. When you compress an image multiple times, different regions compress at different rates. Original, untouched regions compress uniformly. Regions that were edited (pasted in, modified) compress differently because they started at a different compression level.

ELA works by:


Taking the original image
Re-saving it as JPEG at a known quality (e.g., 90%)
Computing the pixel-by-pixel difference between original and re-compressed version
Amplifying that difference (×15) so it's visible


What you see:


Authentic regions → low difference (they were already compressed)
Tampered regions → high difference (they had different compression history)


Original frame  ─── JPEG compress ──→  Compressed frame
     │                                        │
     └──────────── abs(difference) ───────────┘
                        │
                   × scale factor (15)
                        │
                   ELA heatmap

In this system, ELA is computed per frame and blended 50/50 with the original before feeding into the neural network. This gives the model both the visual content AND the compression artifact map simultaneously.


2. EfficientNet-B4 (Spatial Encoder)

What it is: A convolutional neural network (CNN) that extracts features from each individual video frame.

Background — what CNNs do:

A CNN applies a series of learned filters across an image, progressively building up from simple features (edges, colors) to complex features (textures, shapes, semantic regions). By the final layer, it has extracted a compact numerical representation (a feature vector) that summarizes "what's in this image."

Why EfficientNet-B4 specifically:

EfficientNet is a family of CNNs designed with a key idea: instead of just making a network wider (more filters) or deeper (more layers), scale all three dimensions — width, depth, and input resolution — together in a balanced ratio. This gives better accuracy per parameter.

B4 is the 4th variant in the family — a good balance of accuracy and memory usage, suited for an RTX 4050 GPU.

Input frame (224×224×3)
    │
    ▼
Conv Block 0  →  edges, gradients
    │
    ▼
Conv Block 1-2  →  textures, patterns
    │
    ▼
Conv Block 3-5  →  objects, regions
    │
    ▼
Conv Block 6-8  →  high-level semantics
    │
    ▼
Global Average Pool
    │
    ▼
Feature vector (1792 dimensions)

Transfer learning from image model:

Rather than training from scratch, this system loads the EfficientNet-B4 weights from a previously trained image forgery detector. The reasoning: forgery artifacts (compression boundaries, noise patterns, edge inconsistencies) look the same in both images and video frames. The early-layer weights that learned to spot these artifacts are directly reused.

Early layers (blocks 0–4) are frozen — their weights don't change during video training. Later layers fine-tune on video data. This prevents losing the knowledge already learned while adapting to the video domain.


3. Bidirectional GRU (Temporal Head)

What it is: A recurrent neural network that processes the sequence of per-frame features to find cross-frame inconsistencies.

Why we need this for video:

A CNN sees one frame at a time. It cannot ask: "Does frame 30 make sense given what happened in frame 20?" Forged videos often pass frame-level inspection but fail sequence-level inspection — the motion, lighting, or facial expression at the edit boundary doesn't smoothly follow from the previous frames.

What a GRU is:

GRU (Gated Recurrent Unit) is a type of RNN (Recurrent Neural Network). Unlike a standard feedforward network, a GRU has a hidden state — a memory that carries information from one time step to the next.

At each step it decides:


Update gate: How much of the old memory to keep
Reset gate: How much of the old memory to throw away


This allows it to remember long-range patterns — e.g., "the lighting has been consistent for 40 frames but suddenly changed at frame 41."

Why bidirectional:

A standard GRU processes frames left-to-right (past → future). A bidirectional GRU runs two GRUs simultaneously — one forward, one backward — then concatenates their outputs. This means every frame is analyzed in context of both what came before AND what comes after. A splice point looks suspicious from both directions.

Frame 1  Frame 2  Frame 3  ...  Frame 64
   │        │        │               │
   ▼        ▼        ▼               ▼
[Forward GRU  →  →  →  →  →  →  →  →]
[← ← ← ← ← ← ← ←  Backward GRU     ]
   │        │        │               │
   concat(forward, backward) per step
   │
   ▼
Mean pool across all 64 steps
   │
   ▼
Linear → single probability (REAL/FAKE)


4. Optical Flow

What it is: A technique for measuring how pixels move between consecutive frames.

How it works:

Optical flow computes, for every pixel in frame N, the velocity vector (dx, dy) that describes where that pixel moved to in frame N+1. Dense optical flow (Farneback algorithm, used here) does this for every pixel simultaneously.

Frame N          Frame N+1
┌─────────┐      ┌─────────┐
│  pixel  │  →   │  pixel  │
│  at     │      │  at     │
│  (x, y) │      │  (x+dx, │
│         │      │   y+dy) │
└─────────┘      └─────────┘
         flow = (dx, dy)

Why it detects forgeries:

In a real, continuously recorded video, motion is smooth and physically consistent. When a clip is spliced or an object is inserted:


The motion field suddenly changes at the splice boundary — there's an inconsistent "jump"
Inserted objects may move independently from the background motion
Deepfake faces may have subtly wrong motion relative to the head/body


This system computes the standard deviation of flow magnitude across each frame transition. High std dev = chaotic, inconsistent motion = suspicious.

Frames where the flow score exceeds 2.5 standard deviations above the mean are flagged as splice candidates in the output report.


5. Transfer Learning

What it is: Reusing a model trained on one task as a starting point for a related task.

The intuition:

Training a deep neural network requires millions of examples and hours of GPU time. But networks trained on large datasets learn general features that are useful across many tasks. The lower layers of any image-trained CNN learn universal features: edges, corners, textures, gradients. These are just as useful for forgery detection as for any other visual task.

Transfer learning strategy in this system:

Image Forgery Model (trained on CASIA v2.0)
    │
    │  copy weights
    ▼
Video Model Spatial Encoder
    │
    │  freeze blocks 0-4    ← low-level features preserved
    │  fine-tune blocks 5-8 ← adapt to video domain
    │  train temporal GRU   ← new module for sequence reasoning
    ▼
Video Forgery Model

This means the video model starts already knowing how to detect ELA artifacts, compression boundaries, and noise inconsistencies — it only needs to learn the temporal patterns on top.


6. Focal Loss

What it is: A modified loss function that makes the model focus on hard, misclassified examples.

The problem with standard Binary Cross-Entropy (BCE):

In forgery detection, many examples are "easy" — the model quickly learns to classify obvious fakes and obvious real videos with high confidence. Standard BCE treats all examples equally. This means training is dominated by easy examples that contribute little to learning.

How Focal Loss fixes this:

Focal Loss adds a modulating factor (1 - p_t)^γ to BCE:

Focal Loss = -(1 - p_t)^γ × log(p_t)

where:
  p_t  = model's predicted probability for the correct class
  γ    = focusing parameter (we use γ=2)


When the model is correct and confident (p_t → 1): (1-p_t)^2 → 0 — the loss is nearly zero. Easy examples contribute very little.
When the model is wrong or uncertain (p_t → 0): (1-p_t)^2 → 1 — the loss is full strength. Hard examples get full attention.


This naturally shifts training focus toward the ambiguous, difficult-to-classify cases — exactly where the model needs improvement.


System Architecture

┌──────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                               │
│                    (any format: mp4, avi…)                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FRAME SAMPLING                                │
│         64 frames extracted at uniform intervals                 │
│         Captures beginning, middle, and end of video            │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FACE DETECTION                                 │
│         OpenCV Haar Cascade detects faces                        │
│         ✓ Face found  →  crop face region (+ 30% padding)       │
│         ✗ No face     →  use full frame (graceful fallback)      │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ELA BLEND                                     │
│         Each frame → JPEG compress → compute difference          │
│         Blend: 50% original + 50% ELA heatmap                   │
│         Result: model sees content + artifact map together       │
└─────────────────────────┬────────────────────────────────────────┘
                          │ 64 × (224×224×3) frames
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│              EFFICIENTNET-B4 SPATIAL ENCODER                     │
│         Processes each of the 64 frames independently            │
│         Frozen early layers (low-level features)                 │
│         Fine-tuned later layers (forgery-specific features)      │
│         Output: 64 × (1792-d) feature vectors                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │ sequence: (64, 1792)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL GRU TEMPORAL HEAD                     │
│         Projects 1792-d → 512-d per frame                       │
│         Forward GRU + Backward GRU run in parallel               │
│         Mean pool across 64 time steps                           │
│         Linear classifier → single logit                         │
└─────────────────────────┬────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   MODEL SCORE (sigmoid)       TEMPORAL ANALYSIS
   FAKE probability            (parallel, independent)
            │                    Optical flow scores
            │                    Frame diff scores
            ▼                    Splice candidates
   + splice boost if
     candidates > threshold
            │
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FINAL VERDICT                             │
│              REAL / FAKE  +  Confidence %                        │
│              + Temporal Report (splice frame indices)            │
└──────────────────────────────────────────────────────────────────┘


Project Structure

video-forgery-detection/
│
├── video_forgery_detector.py   # Main inference script — run this on any video
├── train_video.py              # Training pipeline (FaceForensics++ dataset)
├── requirements.txt            # pip dependencies
├── README.md                   # This file
├── LICENSE
│
├── models/
│   └── video_model.py          # EfficientNet-B4 + GRU model definition
│
├── utils/
│   ├── ela_video.py            # ELA computation per frame
│   ├── temporal.py             # Optical flow, frame diff, splice detection
│   └── face_utils.py           # Face detection + graceful fallback
│
└── datasets/
    └── ff_dataset.py           # FaceForensics++ PyTorch Dataset class


Setup

bashgit clone https://github.com/YOUR_USERNAME/video-forgery-detection.git
cd video-forgery-detection
pip install -r requirements.txt

Requirements: Python 3.9+, CUDA 12.x recommended (tested on RTX 4050 / 6GB VRAM)


Dataset

This system is designed for FaceForensics++ — the standard benchmark for video forgery detection research. Request access from their official repository (free for academic/research use).

FaceForensics++ contains:


Real videos: YouTube source recordings
Deepfakes: Neural face-swap
Face2Face: Expression transfer
FaceSwap: Geometry-based face swap
NeuralTextures: Neural rendering
FaceShifter: Identity-preserving face reenactment


Expected directory structure:

data/FaceForensics/
  original_sequences/youtube/c23/videos/          ← REAL
  manipulated_sequences/Deepfakes/c23/videos/
  manipulated_sequences/Face2Face/c23/videos/
  manipulated_sequences/FaceSwap/c23/videos/
  manipulated_sequences/NeuralTextures/c23/videos/
  manipulated_sequences/FaceShifter/c23/videos/

c23 refers to the medium-compression version — best balance of visual quality and training signal.


Training

bash# With transfer learning from image forgery model (recommended)
python train_video.py \
  --data_root data/FaceForensics \
  --output_dir checkpoints/video \
  --image_model checkpoints/image/best_model.pt \
  --epochs 30 \
  --batch_size 4 \
  --num_frames 64

# From scratch (no image model)
python train_video.py \
  --data_root data/FaceForensics \
  --output_dir checkpoints/video \
  --epochs 30

# Resume interrupted training
python train_video.py --resume checkpoints/video/best_model.pt

RTX 4050 (6GB VRAM) tips:

SettingValueWhy--batch_size4Fits in 6GB with frames--grad_accum4Effective batch = 16AMPauto-onHalves memory, 2× speedFrozen blocks0–4Saves memory + prevents overfitting


Inference

bashpython video_forgery_detector.py \
  --video path/to/video.mp4 \
  --model checkpoints/video/best_model.pt \
  --output result.json

Sample output:

json{
  "verdict": "FAKE",
  "confidence": 91.3,
  "model_score": 0.8834,
  "face_coverage": 87.5,
  "frames_analyzed": 64,
  "temporal": {
    "splice_count": 3,
    "splice_candidates": [12, 13, 47],
    "max_flow_score": 14.22,
    "max_diff_score": 8.91
  }
}

Reading the output:

FieldMeaningverdictFinal decision: REAL or FAKEconfidenceHow certain the model is (%)model_scoreRaw neural network probability (0–1)face_coverage% of frames where a face was detectedsplice_candidatesFrame indices flagged by optical flow analysissplice_countNumber of suspicious transition points


Ethical Use

This tool is built for research and forensic analysis only.


Do not use to falsely accuse individuals of appearing in manipulated media
Do not use to verify or authenticate non-consensually created deepfakes
Model confidence is not ground truth — treat output as forensic evidence, not proof
Applicable laws: IT Act 2000 §66 (India), GDPR (EU), state deepfake laws (US)


No detection system is 100% accurate. High-quality deepfakes and professionally edited content may evade detection. Always use alongside other verification methods.


Related

This project extends Image Forgery Detection — the EfficientNet-B4 weights transfer directly into this pipeline.


License

MIT License — see LICENSE for details.
