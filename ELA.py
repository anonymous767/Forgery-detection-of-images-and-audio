from pathlib import Path
import cv2
import numpy as np
import os #ability to "talk" to your computer's Operating System
#   ONLY CHANGE THIS LINE
MY_IMAGE = r"C:\miniprojectdataset\CASIA2\Tp\Tp_D_CRN_S_N_cha00071_art00092_11783.jpg"
# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
def load_image(file_path):#file path is a placeholder for the actual path to the image file you want to load. When you call the function, you will replace file_path with the actual path to your image file.
    path = Path(file_path)
    # Does the file exist
    if not path.exists():
        return {"success": False, "error": "File not found"}
    # is it a supported format
    if path.suffix.lower() not in SUPPORTED_FORMATS: # .suffix to check the fomat and .lower to make them lowercase for comparison
        return {"success": False, "error": f"Format {path.suffix} not supported"}
    # try to load it
    try:
        # Load the image — gives you a 3D numpy array
        image = cv2.imread(file_path)

        if image is None:
            return {"success": False, "error": "Could not read image — file may be corrupted"}

        # Get basic info
        height  = image.shape[0]   # number of rows (pixels tall)
        width   = image.shape[1]   # number of columns (pixels wide)
        channels = image.shape[2]  # number of color channels (3 = BGR)
        return {
            "success":   True,
            "image":     image,           # the numpy array
            "height":    height,
            "width":     width,
            "channels":  channels,
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024, # to conver size to KB
            "format":    path.suffix.lower()
        }
    except Exception as e: #If you were writing a calculator app and a user tried to divide by zero, using this statement means your app stays open and says "Cannot divide by zero" instead of simply disappearing or showing a "Not Responding" window.
        return {"success": False, "error": str(e)}

#  Run it and print results
result = load_image(MY_IMAGE)

if result["success"]:
    print(f"  Image loaded successfully!")
    print(f"   File name : {result['file_name']}")
    print(f"   Size      : {result['file_size_kb']:.1f} KB")
    print(f"   Dimensions: {result['width']} x {result['height']} pixels")
    print(f"   Channels  : {result['channels']} (BGR)")
    print(f"   Array shape: {result['image'].shape}")
    print(f"   Top-left pixel (BGR): {result['image'][0, 0]}")
else:
    print(f" Failed: {result['error']}")


def run_ela(image, quality=90):# this reduces the quality of the image to 90% and then compares it to the original image to see if there are any differences. If there are significant differences, it may indicate that the image has been edited or manipulated.

    temp_path = "temp_ela.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])#creating a temp file and writing the image to it with quality 90%
    recompressed = cv2.imread(temp_path)#it for comaparing the image form the og file, recompressed what does is it takes the temp file and loads it back into memory as a numpy array. This allows us to compare the original image with the recompressed version to see if there are any differences that may indicate manipulation.and cv2 .imread is just opeing the file in opencv library and loading it as a numpy array for processing.
    os.remove(temp_path)#The line os.remove(temp_path) is the "cleanup" step. It tells the operating system to permanently delete the file specified by the variable temp_path.
    difference = cv2.absdiff(image, recompressed) 
    ela_map = difference.mean(axis=2)#This line of code takes the 3D color information (Blue, Green, Red) and flattens it into a 2D grayscale map. 
    ela_normalized = ela_map / 255.0 #this line This line of code performs Normalization. It converts the pixel values from "computer-sized" numbers into "math-sized" numbers
     #0 = Pitch Black255 = Pure White
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
    ela_scaled    = (ela_map * 255).astype(np.uint8)#unit8 to save or display an image. and This converts the numbers from "decimals" back into "integers"#
    ela_amplified = cv2.convertScaleAbs(ela_scaled, alpha=10)#acts as a nob multiplies decimal with 10 to make it visible compare with og imag 

    # Apply color map — Blue = clean, Red = suspicious
    heatmap = cv2.applyColorMap(ela_amplified, cv2.COLORMAP_JET)

    output_name = f"ELA_RESULT_{original_file_name}"
    cv2.imwrite(output_name, heatmap)
    return output_name
# Only run ELA if image loaded successfully
if result["success"]:


    print("\nRunning ELA...")
    ela = run_ela(result["image"])

    print(f"   Score      : {ela['score']:.3f}  (0.0 = clean, 1.0 = fake)")
    print(f"   Mean error : {ela['mean_error']:.4f}")
    print(f"   Std error  : {ela['std_error']:.4f}")

    heatmap_file = save_heatmap(ela["ela_map"], result["file_name"])
    print(f"\n  Heatmap saved: {heatmap_file}")
    print(  "   Open this file — Blue = clean | Red = suspicious")

    print("\n" + "=" * 40)
    if ela["suspicious"]:
        print("    VERDICT: LOOKS SUSPICIOUS")
    else:
        print("    VERDICT: LOOKS CLEAN")
    print(f"   Risk Score : {ela['score']:.2f} / 1.00")
    print("=" * 40)

    #these are F-string Format Specifiers. They control how many numbers appear after the decimal point so your report doesn't look messy with long numbers like 0.4582937482
    # 3f/4f means "show 3/4 digits after the decimal point" and .2f means "show 2 digits after the decimal point". You can adjust these numbers to show more or fewer digits as needed.
    #*40 is for border in terminal 
    """
    What is Ela and how does it work 
    ans: ela karta kya hai vo image ko compress karta hai aur phir compressed image ko og image se comapre karta hai 
    aur agar compressed image me koi difference hai to vo difference ko ela map me show karta hai 
     aur us ela map ko heatmap me convert karta hai jisme blue color clean areas ko show karta hai aur red color suspicious areas ko show karta hai 
     aur us heatmap ke basis pe ek score calculate karta hai jo 0 se 1 ke beech hota hai jisme 0 matlab clean aur 1 matlab fake 
     aur agar score 0.35 se zyada hota hai to vo image ko suspicious declare karta hai otherwise clean declare karta hai
    """
    