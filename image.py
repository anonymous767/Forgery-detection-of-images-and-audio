from pathlib import Path
import cv2
import numpy as np
import os #ability to "talk" to your computer's Operating System
import torch # the entire deep learning framework — without this there is no CNN
import timm # library with 300+ pre-trained models — gives us EfficientNet in one line
from torchvision import transforms # tools to prepare image for CNN (resize, normalize etc.)
from PIL import Image # CNN needs images opened with PIL not OpenCV

#   ONLY CHANGE THIS LINE
MY_IMAGE = r"C:\miniprojectdataset\CASIA2\Tp\Tp_D_CRN_S_N_cha00071_art00092_11783.jpg"
# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ==============================================================
#  STEP 1 — LOADER
#  Opens the image file and turns it into a numpy array
#  You already understand this — nothing changed here
# ==============================================================

def load_image(file_path): #file path is a placeholder for the actual path to the image file you want to load. When you call the function, you will replace file_path with the actual path to your image file.

    path = Path(file_path)

    # Does the file exist
    if not path.exists():
        return {"success": False, "error": "File not found"}

    # is it a supported format
    if path.suffix.lower() not in SUPPORTED_FORMATS: # .suffix to check the format and .lower to make them lowercase for comparison
        return {"success": False, "error": f"Format {path.suffix} not supported"}

    # try to load it
    try:
        # Load the image — gives you a 3D numpy array
        image = cv2.imread(file_path)

        if image is None:
            return {"success": False, "error": "Could not read image — file may be corrupted"}

        # Get basic info
        height   = image.shape[0]  # number of rows (pixels tall)
        width    = image.shape[1]  # number of columns (pixels wide)
        channels = image.shape[2]  # number of color channels (3 = BGR)

        return {
            "success":      True,
            "image":        image,           # the numpy array
            "height":       height,
            "width":        width,
            "channels":     channels,
            "file_name":    path.name,
            "file_path":    file_path,       # full path — CNN needs this later
            "file_size_kb": path.stat().st_size / 1024, # to convert size to KB
            "format":       path.suffix.lower()
        }

    except Exception as e: #If you were writing a calculator app and a user tried to divide by zero, using this statement means your app stays open and says "Cannot divide by zero" instead of simply disappearing or showing a "Not Responding" window.
        return {"success": False, "error": str(e)}


# ==============================================================
#  STEP 2 — ELA DETECTION
#  ela karta kya hai vo image ko compress karta hai aur phir
#  compressed image ko og image se compare karta hai
#  aur agar difference hai to ela map me show karta hai
# ==============================================================

def run_ela(image, quality=90): # this reduces the quality of the image to 90% and then compares it to the original image to see if there are any differences. If there are significant differences, it may indicate that the image has been edited or manipulated.

    temp_path = "temp_ela.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality]) #creating a temp file and writing the image to it with quality 90%

    recompressed = cv2.imread(temp_path) #it for comparing the image from the og file, recompressed what does is it takes the temp file and loads it back into memory as a numpy array. This allows us to compare the original image with the recompressed version to see if there are any differences that may indicate manipulation. and cv2.imread is just opening the file in opencv library and loading it as a numpy array for processing.

    os.remove(temp_path) #The line os.remove(temp_path) is the "cleanup" step. It tells the operating system to permanently delete the file specified by the variable temp_path.

    difference = cv2.absdiff(image, recompressed)

    ela_map = difference.mean(axis=2) #This line of code takes the 3D color information (Blue, Green, Red) and flattens it into a 2D grayscale map.

    ela_normalized = ela_map / 255.0 #this line performs Normalization. It converts the pixel values from "computer-sized" numbers into "math-sized" numbers
    #0 = Pitch Black  255 = Pure White

    mean_error = ela_normalized.mean()  # average error across whole image
    std_error  = ela_normalized.std()   # how uneven the errors are

    score = float(min(mean_error * 4.0 + std_error * 2.0, 1.0))
    #mean_error * 4.0: This gives the average error a high priority.
    #std_error * 2.0: This adds extra "suspicion points" if the errors are scattered unevenly.

    return {
        "score":      score,
        "suspicious": score > 0.35,
        "mean_error": float(mean_error),
        "std_error":  float(std_error),
        "ela_map":    ela_normalized
    }


def save_heatmap(ela_map, original_file_name):

    # Scale up small differences so they are visible
    ela_scaled    = (ela_map * 255).astype(np.uint8) #uint8 to save or display an image. and This converts the numbers from "decimals" back into "integers"
    ela_amplified = cv2.convertScaleAbs(ela_scaled, alpha=10) #acts as a knob — multiplies decimal with 10 to make it visible compared with og image

    # Apply color map — Blue = clean, Red = suspicious
    heatmap = cv2.applyColorMap(ela_amplified, cv2.COLORMAP_JET)

    output_name = f"ELA_RESULT_{original_file_name}"
    cv2.imwrite(output_name, heatmap)
    return output_name


# ==============================================================
#  STEP 3 — RED BOX DRAWING (NEW)
#
#  Yeh function ela_map leta hai aur original image pe
#  red rectangles draw karta hai suspicious regions ke around
#
#  Kaise kaam karta hai:
#
#  ela_map ek 2D array hai — har pixel ki suspicion value hai
#  Example:
#  0.01  0.02  0.01  0.71  0.82   ← high values = suspicious
#  0.02  0.01  0.02  0.68  0.91   ← tampered region
#  0.01  0.02  0.01  0.72  0.88   ← tampered region
#
#  Step 1: Threshold — sab kuch 0.5 se upar = white (suspicious)
#                      sab kuch 0.5 se neeche = black (clean)
#  Step 2: Find contours — white blobs ke outlines dhundho
#  Step 3: Draw rectangle — har contour ke around red box
# ==============================================================

def draw_red_boxes(original_image, ela_map):
    """
    ela_map ke basis pe original image pe red boxes draw karta hai
    jahan bhi tampering detect hui wahan red rectangle aata hai
    """

    # Step 1: Image ki copy banao — original modify mat karo
    output_image = original_image.copy()
    # .copy() matlab ek bilkul alag copy — original safe rehta hai

    # Step 2: ela_map ko 0-255 range mein scale karo
    # ela_map mein values 0.0-1.0 hain
    # OpenCV threshold ke liye 0-255 chahiye
    ela_scaled = (ela_map * 255).astype(np.uint8)

    # Step 3: Amplify — chhoti differences ko bada karo taaki dikhein
    ela_amplified = cv2.convertScaleAbs(ela_scaled, alpha=10)
    # alpha=10 matlab har value ko 10 se multiply karo
    # 0.05 → 0.5 → after scaling = visible

    # Step 4: Threshold — binary image banao (sirf black aur white)
    # 127 = midpoint of 0-255
    # Sab kuch 127 se upar → 255 (white) = suspicious region
    # Sab kuch 127 se neeche → 0 (black) = clean region
    _, binary = cv2.threshold(ela_amplified, 127, 255, cv2.THRESH_BINARY)
    # _ matlab hum pehli return value ignore kar rahe hain (threshold value)
    # binary = black/white image jisme white = suspicious

    # Step 5: Noise hatao using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # kernel = 5x5 grid of 1s
    # Morphological operations is grid ko image pe slide karte hain

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # MORPH_OPEN = erosion phir dilation
    # Chhote white dots hatata hai jo sirf noise hain
    # Real suspicious regions ko rakhta hai

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # MORPH_CLOSE = dilation phir erosion
    # Suspicious regions ke andar ke holes fill karta hai
    # Blobs ko solid banata hai taaki contours clean hon

    # Step 6: Contours dhundho — white blobs ke outlines
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = list of all white blobs
    # RETR_EXTERNAL = sirf outer contours (andar ke holes nahi)
    # CHAIN_APPROX_SIMPLE = contour points compress karo (memory bachao)

    # Step 7: Har contour ke around red rectangle draw karo
    boxes_drawn = 0
    min_area    = 500
    # min_area = minimum pixel area to draw box
    # 500 pixels se chhota = probably noise, ignore karo

    for contour in contours:
        # contour = ek suspicious region ki boundary points

        area = cv2.contourArea(contour)
        # contourArea = is contour ke andar kitne pixels hain

        if area > min_area:
            # Sirf bade enough regions pe box draw karo

            x, y, w, h = cv2.boundingRect(contour)
            # boundingRect = smallest rectangle jo is contour ko contain kare
            # x, y = top-left corner ka position
            # w    = width (pixels wide)
            # h    = height (pixels tall)

            # Red rectangle draw karo
            cv2.rectangle(
                output_image,      # image jis pe draw karna hai
                (x, y),            # top-left corner
                (x + w, y + h),    # bottom-right corner
                (0, 0, 255),       # color: BGR format mein (0,0,255) = RED
                3                  # line thickness = 3 pixels
            )
            # BGR kyun: OpenCV BGR order use karta hai (Blue Green Red)
            # (0, 0, 255) matlab Blue=0, Green=0, Red=255 = pure red

            # Box ke upar label likhte hain
            cv2.putText(
                output_image,
                "SUSPICIOUS",              # text jo likhna hai
                (x, y - 10),              # position — box ke 10 pixels upar
                cv2.FONT_HERSHEY_SIMPLEX, # font style
                0.7,                       # font size
                (0, 0, 255),              # color: RED
                2                          # text thickness
            )

            boxes_drawn += 1
            # += 1 matlab boxes_drawn = boxes_drawn + 1

    return output_image, boxes_drawn
    # output_image = original image with red boxes drawn on it
    # boxes_drawn  = kitne boxes draw kiye gaye


# ==============================================================
#  STEP 4 — DEEP CNN (EfficientNet-B4)
#
#  CNN kya hai:
#  EfficientNet-B4 ek deep neural network hai jisme
#  hundreds of layers hain. Har layer pehle wali se
#  zyada complex patterns dhundhti hai:
#
#  Layer 1  → basic edges aur lines
#  Layer 2  → corners aur curves
#  Layer 3  → textures aur patterns
#  Layer 4  → unnatural smoothness
#  Layer 5  → blending boundaries
#  Layer 6  → compression artifacts
#  Layer 7+ → woh cheezein jo human dekh nahi sakta
#  Final    → REAL ya FAKE probability
#
#  Pre-trained matlab:
#  Google ne isko 1.2 million images pe train kiya hai
#  pehle se. Hum sirf download karke use karte hain.
#  pretrained=True likhne se vo saari learning mil jaati hai.
#
#  NOTE: Abhi pretrained=True use kar rahe hain ImageNet weights se
#  Jab CASIA dataset pe train karoge tab forgery_model.pth
#  load hoga jo actually forgery samjhega
# ==============================================================

def load_cnn_model():

    print("   Loading EfficientNet-B4...")

    # Check karo GPU available hai ya nahi
    # cuda = your NVIDIA RTX 4050
    # agar cuda available hai to GPU use karo (fast)
    # warna CPU use karo (slow but still works)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.device decides WHERE calculations happen — GPU ya CPU
    print(f"   Running on: {device}")

    # EfficientNet-B4 load karo timm library se
    # pretrained=True  = millions of images pe already trained weights use karo
    # num_classes=2    = sirf 2 outputs chahiye: [real score, fake score]
    model = timm.create_model(
        "efficientnet_b4",  # deep CNN architecture ka naam
        pretrained=True,    # pehle se trained weights download karo
        num_classes=2       # output: [real probability, fake probability]
    )

    # Model ko GPU memory mein move karo
    # Jaise file ko hard drive se RAM mein load karte ho
    # yahan model ko CPU se GPU mein move kar rahe hain
    model = model.to(device)

    # eval() training-specific features band karta hai
    # Training mein model randomly neurons band karta hai (Dropout)
    # Prediction mein hum yeh nahi chahte — consistent results chahiye
    model.eval()

    return model, device


def prepare_image_for_cnn(file_path):

    # Yeh values ImageNet dataset ki mean aur std hain
    # EfficientNet ko in exact values se normalize ki hui images pe train kiya tha
    # Hum SAME values use karte hain — warna model confused ho jaata hai
    imagenet_mean = [0.485, 0.456, 0.406]  # average R, G, B across ImageNet
    imagenet_std  = [0.229, 0.224, 0.225]  # spread of R, G, B across ImageNet

    # transforms.Compose multiple steps ko ek pipeline mein chain karta hai
    # Jaise assembly line mein ek ke baad ek step hota hai
    transform = transforms.Compose([

        transforms.Resize((380, 380)),
        # EfficientNet-B4 ko exactly 380x380 size chahiye
        # Isse chota ya bada doge to math break ho jaata hai

        transforms.ToTensor(),
        # PIL image ko PyTorch tensor mein convert karta hai
        # Tensor = numpy array jaisa but GPU pe kaam karta hai
        # Saath mein pixel values 0-255 se 0.0-1.0 mein convert karta hai

        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
        # Formula: normalized = (pixel - mean) / std
        # Yeh step ensure karta hai ki numbers model ki expectation se match karein
    ])

    # PIL se image open karo RGB format mein
    # CNN ko RGB chahiye (Red Green Blue)
    # OpenCV BGR deta hai isliye yahan PIL use karte hain
    image_pil = Image.open(file_path).convert("RGB")

    # Saare transformations apply karo
    tensor = transform(image_pil)

    # Batch dimension add karo
    # Model expect karta hai: (batch_size, channels, height, width)
    # Hamari image: (3, 380, 380)
    # unsqueeze(0) ke baad: (1, 3, 380, 380) = ek image ka batch
    tensor = tensor.unsqueeze(0)
    # Bilkul jaise ek roti ko packet mein daalna — same roti, sirf packaging alag

    return tensor


def run_cnn(file_path, model, device):

    # Image ko CNN format mein prepare karo
    tensor = prepare_image_for_cnn(file_path)

    # Tensor ko GPU pe move karo — model bhi wahan hai
    # Dono same jagah hone chahiye warna error aata hai
    tensor = tensor.to(device)

    with torch.no_grad():
        # torch.no_grad() matlab: hum training nahi kar rahe, sirf predict kar rahe hain
        # Training mein PyTorch har calculation track karta hai (gradients ke liye)
        # Prediction mein yeh zaroori nahi — is liye band karte hain
        # Result: faster + less GPU memory use

        # YAHAN SARA DEEP LEARNING HOTA HAI
        # Tumhari image EfficientNet-B4 ke hundreds of layers se guzarti hai
        # Sab kuch tumhare RTX 4050 pe milliseconds mein
        output = model(tensor)
        # output shape: (1, 2)
        # output[0][0] = real ka raw score (logit kehte hain)
        # output[0][1] = fake ka raw score (logit kehte hain)
        # Example: [-1.2, 2.4] — yeh directly percentages nahi hain

        # Softmax raw numbers ko probabilities mein convert karta hai
        # Dono numbers ko 0-1 ke beech laata hai aur dono ka sum = 1.0
        # Example: [-1.2, 2.4] → [0.08, 0.92] = 8% real, 92% fake
        probabilities = torch.softmax(output, dim=1)

        # .item() PyTorch tensor ko regular Python float mein convert karta hai
        # Bina .item() ke yeh number PyTorch ke andar stuck rehta hai
        fake_prob = probabilities[0][1].item()  # index 1 = fake probability
        real_prob = probabilities[0][0].item()  # index 0 = real probability

    return {
        "score":            float(fake_prob),   # yahi hamara CNN score hai
        "suspicious":       fake_prob > 0.50,   # 50% se zyada = suspicious
        "fake_probability": float(fake_prob),
        "real_probability": float(real_prob)
    }


# ==============================================================
#  STEP 5 — FUSION
#  ELA aur CNN dono ke scores ko mila ke ek final verdict deta hai
#
#  Dono kyun use karte hain?
#  ELA  → batata hai KAHAN tampering hui (heatmap + red boxes)
#          lekin heavily compressed images pe false positive de sakta hai
#  CNN  → batata hai KYA image fake hai overall
#          lekin nahi batata KAHAN tampering hui
#  Dono saath → ek doosre ki weakness cover karte hain
# ==============================================================

def fuse_scores(ela_result, cnn_result):

    ela_score = ela_result["score"]
    cnn_score = cnn_result["score"]

    # Weighted average
    # CNN ko 65% weight — zyada trustworthy hai overall
    # ELA ko 35% weight — supporting evidence ki tarah
    risk_score = (cnn_score * 0.65) + (ela_score * 0.35)

    # Override rule
    # Agar koi ek module bahut zyada confident hai to automatically FAKE
    override        = False
    override_reason = ""

    if cnn_score > 0.85:
        override        = True
        override_reason = f"CNN bahut confident hai forgery ke baare mein ({cnn_score:.0%})"

    if ela_score > 0.80:
        override        = True
        override_reason = f"ELA ne severe tampering detect ki ({ela_score:.2f})"

    # Kitne modules ne suspicious flag kiya
    # Python mein True = 1 aur False = 0
    # isliye sum([True, False]) = 1 aur sum([True, True]) = 2
    suspicious_count = sum([
        ela_result["suspicious"],
        cnn_result["suspicious"]
    ])

    # Final decision
    if override or suspicious_count >= 2:
        is_fake = True
    else:
        is_fake = risk_score > 0.50

    # Confidence = kitna door hai score 0.5 se
    # 0.5 = bilkul uncertain, 1.0 = bilkul certain
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
#  MAIN — Sab kuch connect karta hai
# ==============================================================

# Step 1: Load the image
result = load_image(MY_IMAGE)

if result["success"]:
    print(f"\n  Image loaded successfully!")
    print(f"   File name : {result['file_name']}")
    print(f"   Size      : {result['file_size_kb']:.1f} KB")
    print(f"   Dimensions: {result['width']} x {result['height']} pixels")
    print(f"   Channels  : {result['channels']} (BGR)")
    print(f"   Array shape: {result['image'].shape}")
    print(f"   Top-left pixel (BGR): {result['image'][0, 0]}")
else:
    print(f" Failed: {result['error']}")


if result["success"]:

    # ── ELA OUTPUT ───────────────────────────────────────────
    print("\n" + "=" * 45)
    print("   🔬 ELA RESULT (Error Level Analysis)")
    print("=" * 45)

    ela = run_ela(result["image"])

    print(f"   Score      : {ela['score']:.3f}  (0.0 = clean, 1.0 = fake)")
    print(f"   Mean error : {ela['mean_error']:.4f}")
    print(f"   Std error  : {ela['std_error']:.4f}")

    #these are F-string Format Specifiers. They control how many numbers appear after the decimal point so your report doesn't look messy with long numbers like 0.4582937482
    # 3f/4f means "show 3/4 digits after the decimal point" and .2f means "show 2 digits after the decimal point". You can adjust these numbers to show more or fewer digits as needed.
    #*45 is for border in terminal

    # Save heatmap
    heatmap_file = save_heatmap(ela["ela_map"], result["file_name"])
    print(f"\n   Heatmap saved : {heatmap_file}")
    print(  "   Open this file — Blue = clean | Red = suspicious")

    # Draw red boxes on image and save
    # draw_red_boxes function ela_map leta hai
    # suspicious regions dhundta hai aur red rectangles draw karta hai
    image_with_boxes, boxes_count = draw_red_boxes(
        result["image"],   # original image
        ela["ela_map"]     # suspicion map from ELA
    )
    boxes_file = f"BOXES_{result['file_name']}"
    cv2.imwrite(boxes_file, image_with_boxes)
    # cv2.imwrite saves the image to disk

    print(f"   Boxes saved   : {boxes_file}")
    print(f"   Red boxes drawn: {boxes_count} suspicious region(s) found")

    if ela["suspicious"]:
        print("\n   ELA Status : ⚠️  SUSPICIOUS")
    else:
        print("\n   ELA Status : ✅ Clean")

    # ── CNN OUTPUT ───────────────────────────────────────────
    print("\n" + "=" * 45)
    print("   🧠 CNN RESULT (EfficientNet-B4)")
    print("=" * 45)
    # Note: pehli baar chalane pe model download hoga (~70MB)
    # Yeh sirf ek baar hota hai — uske baad instantly load hota hai

    model, device = load_cnn_model()
    cnn = run_cnn(result["file_path"], model, device)

    print(f"\n   Score      : {cnn['score']:.3f}  (0.0 = real, 1.0 = fake)")
    print(f"   Real prob  : {cnn['real_probability']:.1%}")
    print(f"   Fake prob  : {cnn['fake_probability']:.1%}")

    if cnn["suspicious"]:
        print("   CNN Status : ⚠️  SUSPICIOUS")
    else:
        print("   CNN Status : ✅ Clean")

    # ── FINAL VERDICT ────────────────────────────────────────
    final = fuse_scores(ela, cnn)

    print("\n" + "=" * 45)
    if final["is_fake"]:
        print("   🚨 VERDICT: LIKELY TAMPERED / FAKE")
    else:
        print("   ✅ VERDICT: LIKELY REAL / AUTHENTIC")

    print(f"   Confidence : {final['confidence']:.0%}")
    print(f"   Risk Score : {final['risk_score']:.2f} / 1.00")

    if final["override"]:
        print(f"   ⚡ Override: {final['override_reason']}")

    print("\n   Module Breakdown:")
    print(f"   ELA : {ela['score']:.2f}  {'⚠️  Suspicious' if ela['suspicious'] else '✅ Clean'}")
    print(f"   CNN : {cnn['score']:.2f}  {'⚠️  Suspicious' if cnn['suspicious'] else '✅ Clean'}")

    print("\n   Output Files Saved:")
    print(f"   📦 {boxes_file}   ← open this — red boxes on suspicious regions")
    print(f"   🗺️  {heatmap_file} ← open this — blue/red heatmap")
    print("=" * 45)

    """
    What is ELA and how does it work
    ans: ela karta kya hai vo image ko compress karta hai aur phir compressed image ko og image se compare karta hai
    aur agar compressed image me koi difference hai to vo difference ko ela map me show karta hai
    aur us ela map ko heatmap me convert karta hai jisme blue color clean areas ko show karta hai aur red color suspicious areas ko show karta hai
    aur us heatmap ke basis pe ek score calculate karta hai jo 0 se 1 ke beech hota hai jisme 0 matlab clean aur 1 matlab fake
    aur agar score 0.35 se zyada hota hai to vo image ko suspicious declare karta hai otherwise clean declare karta hai

    What is Red Box Drawing and how does it work
    ans: draw_red_boxes function ela_map leta hai
    ela_map ko threshold karta hai — 0.5 se upar = white (suspicious), neeche = black (clean)
    phir white regions ke contours (outlines) dhundta hai
    har contour ke around ek red rectangle draw karta hai original image pe
    box ke upar "SUSPICIOUS" text bhi likhta hai
    500 pixels se chhote regions ignore karta hai (noise hote hain)
    result: original image with red boxes = BOXES_filename.jpg

    What is Deep CNN and how does it work
    ans: CNN ek deep neural network hai jisme hundreds of layers hain
    har layer pehle wali se zyada complex patterns dhundhti hai
    EfficientNet-B4 Google ne 1.2 million images pe train kiya tha
    pretrained=True se vo saari learning hume mil jaati hai bina training ke
    image 380x380 pe resize hoti hai, normalize hoti hai, tensor banti hai
    phir model ke through pass hoti hai aur [real, fake] probability milti hai
    softmax se raw numbers probabilities mein convert hote hain
    50% se zyada fake probability = image suspicious hai

    How ELA and CNN work together (Fusion)
    ans: ELA batata hai KAHAN tampering hui — heatmap + red boxes se dikhta hai
    CNN batata hai KYA image fake hai — overall verdict deta hai
    dono ke scores ko mila ke ek final risk score banate hain
    CNN ko 65% weight dete hain kyunki zyada reliable hai
    ELA ko 35% weight dete hain supporting evidence ki tarah
    agar koi ek 85% se zyada confident ho to override ho jaata hai
    dono suspicious hon to automatically FAKE declare hota hai
    """