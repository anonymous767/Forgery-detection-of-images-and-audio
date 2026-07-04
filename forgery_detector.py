

from pathlib import Path
import cv2
import numpy as np
import os
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image
import PIL.ExifTags

# =============================================
#   ONLY CHANGE THIS LINE
# =============================================
MY_IMAGE   = r"C:\miniprojectdataset\CASIA2\Tp\Tp_D_CRN_S_N_cha00071_art00092_11783.jpg"
MODEL_PATH = "forgery_model.pth"
# =============================================

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ==============================================================
#  STEP 1 — LOADER
# ==============================================================

def load_image(file_path):

    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": "File not found"}

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return {"success": False, "error": f"Format {path.suffix} not supported"}

    try:
        image = cv2.imread(file_path)
        if image is None:
            return {"success": False, "error": "Could not read image"}

        return {
            "success":      True,
            "image":        image,
            "file_name":    path.name,
            "file_path":    file_path,
            "width":        image.shape[1],
            "height":       image.shape[0],
            "file_size_kb": path.stat().st_size / 1024
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ==============================================================
#  STEP 2 — ELA
#  Finds WHERE tampering happened
#  Pure math — no GPU needed, runs instantly on any laptop
# ==============================================================

def run_ela(image, quality=90):

    temp_path = "temp_ela.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    recompressed = cv2.imread(temp_path)
    os.remove(temp_path)

    difference     = cv2.absdiff(image, recompressed)
    ela_map        = difference.mean(axis=2)
    ela_normalized = ela_map / 255.0

    mean_error = ela_normalized.mean()
    std_error  = ela_normalized.std()
    score      = float(min(mean_error * 4.0 + std_error * 2.0, 1.0))

    return {
        "score":      score,
        "suspicious": score > 0.35,
        "mean_error": float(mean_error),
        "std_error":  float(std_error),
        "ela_map":    ela_normalized
    }


def save_heatmap(ela_map, file_name):
    ela_scaled    = (ela_map * 255).astype(np.uint8)
    ela_amplified = cv2.convertScaleAbs(ela_scaled, alpha=10)
    heatmap       = cv2.applyColorMap(ela_amplified, cv2.COLORMAP_JET)
    output_name   = f"ELA_HEATMAP_{file_name}"
    cv2.imwrite(output_name, heatmap)
    return output_name


# ==============================================================
#  STEP 3 — RED BOX DRAWING
#  Draws red rectangles around suspicious regions from ELA map
# ==============================================================

def draw_red_boxes(original_image, ela_map):

    output_image  = original_image.copy()
    ela_scaled    = (ela_map * 255).astype(np.uint8)
    ela_amplified = cv2.convertScaleAbs(ela_scaled, alpha=10)

    _, binary = cv2.threshold(ela_amplified, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_drawn = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(output_image, "SUSPICIOUS", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            boxes_drawn += 1

    return output_image, boxes_drawn


# ==============================================================
#  STEP 4 — MOBILENETV2 CNN
#
#  Kyun MobileNetV2:
#  → 3.4M parameters (EfficientNet ke 19M vs)
#  → 224x224 input (EfficientNet ke 380x380 vs)
#  → CPU pe fast — mobile phones ke liye banaya tha
#  → 14MB weight file (EfficientNet ke 70MB vs)
#  → 8GB RAM mein aaram se fit
#
#  Auto-detects GPU or CPU:
#  → GPU mile to cuda use karo (fast)
#  → GPU na mile to CPU use karo (works fine)
# ==============================================================

def load_cnn_model(model_path):

    # Auto-detect GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"   ℹ CPU mode (no GPU found)")

    # Load MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    # weights=None because we load OUR trained weights

    # Same change as in train.py — 2 output classes
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # Check if trained model exists
    if not Path(model_path).exists():
        print(f"\n forgery_model.pth not found!")
        print(f"   Run train.py first to create it")
        return None, device

    # Load YOUR trained weights
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    # map_location=device ensures it loads correctly
    # whether on GPU or CPU

    model = model.to(device)
    model.eval()
    return model, device


def run_cnn(file_path, model, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # MobileNetV2 expects 224x224
        # Much smaller than EfficientNet's 380x380
        # Faster to process on CPU

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Same ImageNet normalization as training
        # MUST match train.py exactly
    ])

    image_pil = Image.open(file_path).convert("RGB")
    tensor    = transform(image_pil).unsqueeze(0).to(device)
    # unsqueeze(0) adds batch dimension
    # .to(device) moves to GPU or CPU

    with torch.no_grad():
        output        = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        fake_prob     = probabilities[0][1].item()
        real_prob     = probabilities[0][0].item()

    return {
        "score":            float(fake_prob),
        "suspicious":       fake_prob > 0.50,
        "fake_probability": float(fake_prob),
        "real_probability": float(real_prob)
    }


# ==============================================================
#  STEP 5 — METADATA CHECK
#  Reads hidden EXIF data from the image file
#  Checks for editing software, missing camera info etc.
#  Instant — no GPU, no math, just reading file data
# ==============================================================

def run_metadata_check(file_path):

    warnings = []
    score    = 0.0

    try:
        img  = Image.open(file_path)
        exif = img._getexif()
        # _getexif() reads hidden metadata from image file
        # Real camera photos always have this
        # Edited images often have it missing or modified

        if exif is None:
            warnings.append("No EXIF metadata — possibly stripped by editing software")
            score += 0.3
            # Real camera photos always have EXIF
            # Missing EXIF = suspicious

        else:
            # Convert numeric tag IDs to readable names
            exif_data = {
                PIL.ExifTags.TAGS.get(tag, tag): value
                for tag, value in exif.items()
            }

            # Check for known editing software
            software = str(exif_data.get("Software", ""))
            editors  = ["Photoshop", "GIMP", "Lightroom",
                        "Snapseed", "PicsArt", "Meitu"]

            for editor in editors:
                if editor.lower() in software.lower():
                    warnings.append(f"Edited with: {software}")
                    score += 0.4
                    break

            # Real photos always have camera make and model
            if "Make" not in exif_data:
                warnings.append("Camera make missing")
                score += 0.15

            if "Model" not in exif_data:
                warnings.append("Camera model missing")
                score += 0.15

            # Check if photo was modified after being taken
            date_taken    = exif_data.get("DateTimeOriginal")
            date_modified = exif_data.get("DateTime")

            if date_taken and date_modified:
                if date_taken != date_modified:
                    warnings.append("File modified after original capture date")
                    score += 0.2

    except Exception as e:
        warnings.append(f"Could not read metadata: {str(e)}")
        score = 0.1

    score = float(min(score, 1.0))

    return {
        "score":      score,
        "suspicious": score > 0.35,
        "warnings":   warnings
    }


# ==============================================================
#  STEP 6 — FUSION
#  Combines all 3 scores into final verdict
#
#  Weights:
#  CNN      = 50% (main classifier)
#  ELA      = 30% (finds where tampering is)
#  Metadata = 20% (supporting evidence)
# ==============================================================

def fuse_scores(ela_result, cnn_result, meta_result):

    ela_score  = ela_result["score"]
    cnn_score  = cnn_result["score"]
    meta_score = meta_result["score"]

    risk_score = (cnn_score  * 0.50) + \
                 (ela_score  * 0.30) + \
                 (meta_score * 0.20)

    # Override — if any module is extremely confident
    override        = False
    override_reason = ""

    if cnn_score > 0.85:
        override        = True
        override_reason = f"CNN very confident ({cnn_score:.0%})"

    if ela_score > 0.80:
        override        = True
        override_reason = f"ELA severe tampering ({ela_score:.2f})"

    # Count suspicious modules
    suspicious_count = sum([
        ela_result["suspicious"],
        cnn_result["suspicious"],
        meta_result["suspicious"]
    ])

    if override or suspicious_count >= 2:
        is_fake = True
    else:
        is_fake = risk_score > 0.50

    confidence = abs(risk_score - 0.5) * 2.0
    if override:
        confidence = max(confidence, 0.85)

    return {
        "is_fake":         is_fake,
        "risk_score":      float(risk_score),
        "confidence":      float(min(confidence, 1.0)),
        "override":        override,
        "override_reason": override_reason
    }


# ==============================================================
#  MAIN
# ==============================================================

def analyze(file_path):

    print("\n" + "=" * 52)
    print("    IMAGE FORGERY DETECTOR")
    print("=" * 52)

    # Load image
    print("\nLoading image...")
    loaded = load_image(file_path)

    if not loaded["success"]:
        print(f" {loaded['error']}")
        return

    print(f"   ✅{loaded['file_name']}")
    print(f"    {loaded['width']} x {loaded['height']} pixels")
    print(f"    {loaded['file_size_kb']:.1f} KB")

    image = loaded["image"]

    # ── ELA ──────────────────────────────────────────────────
    print("\n" + "━" * 52)
    print("    ELA RESULT")
    print("━" * 52)

    ela = run_ela(image)

    print(f"   Score      : {ela['score']:.3f}")
    print(f"   Mean error : {ela['mean_error']:.4f}")
    print(f"   Std error  : {ela['std_error']:.4f}")
    print(f"   Status     : {'  SUSPICIOUS' if ela['suspicious'] else 'Clean'}")

    heatmap_file = save_heatmap(ela["ela_map"], loaded["file_name"])
    print(f"\n     Heatmap → {heatmap_file}")

    image_with_boxes, boxes_count = draw_red_boxes(image, ela["ela_map"])
    boxes_file = f"BOXES_{loaded['file_name']}"
    cv2.imwrite(boxes_file, image_with_boxes)
    print(f"    Boxes   → {boxes_file} ({boxes_count} region(s) marked)")

    # ── CNN ──────────────────────────────────────────────────
    print("\n" + "━" * 52)
    print("   CNN RESULT (MobileNetV2)")
    print("━" * 52)

    print("\n   Loading model...")
    model, device = load_cnn_model(MODEL_PATH)

    if model is None:
        return

    cnn = run_cnn(loaded["file_path"], model, device)

    print(f"   Score      : {cnn['score']:.3f}")
    print(f"   Real prob  : {cnn['real_probability']:.1%}")
    print(f"   Fake prob  : {cnn['fake_probability']:.1%}")
    print(f"   Status     : {' SUSPICIOUS' if cnn['suspicious'] else ' Clean'}")

    # ── METADATA ─────────────────────────────────────────────
    print("\n" + "━" * 52)
    print("    METADATA RESULT")
    print("━" * 52)

    meta = run_metadata_check(loaded["file_path"])

    print(f"   Score      : {meta['score']:.3f}")
    print(f"   Status     : {'  SUSPICIOUS' if meta['suspicious'] else ' Clean'}")

    if meta["warnings"]:
        for w in meta["warnings"]:
            print(f"     {w}")
    else:
        print("   No suspicious metadata found")

    # ── FINAL VERDICT ────────────────────────────────────────
    final = fuse_scores(ela, cnn, meta)

    print("\n" + "=" * 52)
    if final["is_fake"]:
        print("    VERDICT: LIKELY TAMPERED / FAKE")
    else:
        print("    VERDICT: LIKELY REAL / AUTHENTIC")

    print(f"   Confidence : {final['confidence']:.0%}")
    print(f"   Risk Score : {final['risk_score']:.2f} / 1.00")

    if final["override"]:
        print(f"   Override: {final['override_reason']}")

    print("\n   Module Breakdown:")
    print(f"   ELA      : {ela['score']:.2f}  {' Suspicious' if ela['suspicious'] else ' Clean'}")
    print(f"   CNN      : {cnn['score']:.2f}  {' Suspicious' if cnn['suspicious'] else 'Clean'}")
    print(f"   Metadata : {meta['score']:.2f}  {'Suspicious' if meta['suspicious'] else ' Clean'}")

    print("\n   Saved Files:")
    print(f"    {boxes_file}  ← red boxes on tampered regions")
    print(f"     {heatmap_file}  ← blue/red heatmap")
    print("=" * 52 + "\n")


analyze(MY_IMAGE)
