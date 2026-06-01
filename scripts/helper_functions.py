from math import ceil
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import random
from glob import glob
import pandas as pd
import os
import cv2
import numpy as np

# Reading functions 

def collect_kaggle_images(split_dir, label):
    paths = glob(os.path.join(split_dir, label, "*.jpg"))
    return pd.DataFrame({"file_path": paths, "label": label})

# Image processing

def compute_hash(fp):
    try:
        return str(imagehash.phash(Image.open(fp).convert('RGB').resize((128, 128))))
    except:
        return None
    
def resize(img_path, max_size=1000, save_over=True):
    img = Image.open(img_path)
    w, h = img.size

    if max(w, h) <= max_size:
        return False  # No resize needed

    # Compute scale while preserving aspect ratio
    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    resized = img.resize(new_size, Image.LANCZOS)

    if save_over:
        resized.save(img_path)
    else:
        base, ext = os.path.splitext(img_path)
        resized.save(f"{base}_resized{ext}")

    return True  

## Vignetting

def has_vignette_border(img_path, threshold=0.6, img_size=224):
    try:
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)
        
        height, width = img_np.shape

        # Calculate proportional patch sizes, using 10%
        patch_h = ceil(height * 0.10)
        patch_w = ceil(width * 0.10)
        
        corners = [
            img_np[:patch_h, :patch_w],                    # Top-Left
            img_np[:patch_h, -patch_w:],                   # Top-Right
            img_np[-patch_h:, :patch_w],                   # Bottom-Left
            img_np[-patch_h:, -patch_w:]                   # Bottom-Right
        ]

        border_pixels = np.concatenate([c.flatten() for c in corners])
        
        # Count near-black or near-white pixels
        mask = (border_pixels < 10) | (border_pixels > 245)
        return np.mean(mask) > threshold
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

    

def crop_to_detected_circle(img_path, min_shrink=0.6, max_shrink=0.95):
    import cv2
    import numpy as np
    from PIL import Image

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return Image.open(img_path)

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # === Try detecting a circle with Hough method ===
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 2,
            param1=100, param2=30,
            minRadius=int(0.5 * min(h, w)),
            maxRadius=int(0.9 * min(h, w))
        )
    except cv2.error:
        circles = None
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


    # === Fallback to contour circle ===
    if circles is not None:
        x, y, r = circles[0][0]
    else:
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        else:
            # fallback: center crop with preserved aspect ratio
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # === Shrink factor based on radius ===
    r = float(r)
    shrink = max(min_shrink, min(max_shrink, 1.0 - 0.0004 * r**1.2))

    # === Crop size in pixels (preserve ratio) ===
    crop_w = int(w * shrink)
    crop_h = int(h * shrink)

    # === Centered box around (x, y) ===
    x = int(x)
    y = int(y)
    x1 = max(0, x - crop_w // 2)
    y1 = max(0, y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # Adjust x1/y1 if x2/y2 got clipped
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    # Final crop
    cropped = img_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

# def crop_border(img_path):
#     """
#     Efficiently crops circular black or white borders from the image,
#     preserving the aspect ratio.
#     """
#     # Load image
#     img = cv2.imread(img_path)
#     if img is None:
#         return Image.open(img_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     # 1. Detect border type (black or white) using corners
#     corners = [
#         gray[0, 0],
#         gray[0, -1],
#         gray[-1, 0],
#         gray[-1, -1]
#     ]
#     avg_corner = np.mean(corners)
#     border_type = 'black' if avg_corner < 100 else 'white'

#     # 2. Create mask of border (black: low values, white: high values)
#     if border_type == 'black':
#         mask = gray > 30  # keep pixels that are not black
#     else:
#         mask = gray < 225  # keep pixels that are not white

#     # 3. Distance transform from center outwards
#     center_y, center_x = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)

#     # 4. Masked region (non-border), compute max usable radius
#     usable_mask = mask.astype(np.uint8)
#     usable_coords = np.where(usable_mask)
#     usable_dists = dist_from_center[usable_coords]
#     print(usable_dists)
#     if len(usable_dists) == 0:
#         return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # fallback: no crop

#     max_r = np.percentile(usable_dists, 99)  # avoid outliers

#     # 5. Compute box that fits inside max_r and matches aspect ratio
#     img_ratio = w / h
#     box_h = int(2 * max_r / np.sqrt(1 + img_ratio**2))
#     box_w = int(box_h * img_ratio)
#     print(box_h,box_w)
#     # Ensure bounds
#     box_h = min(box_h, h) #- int(0.1*h)
#     box_w = min(box_w, w) #- int(0.1*w)

#     y1 = max(center_y - box_h // 2, 0)
#     y2 = y1 + box_h
#     x1 = max(center_x - box_w // 2, 0)
#     x2 = x1 + box_w

#     print(x1,x2,y1,y2)
#     # 6. Crop and return
#     cropped = img[y1:y2, x1:x2]
#     # cropped = img[x1:x2,y1:y2]

#     return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

## Hair removal 
def remove_hair(img):
    img_bgr = np.array(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    inpainted = cv2.inpaint(img_bgr, hair_mask, 1, cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def scan_corner_to_center(gray, start, threshold=15, max_steps=300):
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    # print(w,h)
    x, y = start
    ref = gray[y, x]
    path = []

    for i in range(max_steps):
        # Ensure within bounds
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        val = gray[y, x]
        # print(x,y)
        # print(ref, val)
        if abs(int(val) - int(ref)) > threshold:
            break
        path.append((x, y))

        # Step toward center
        dx = 1 if x < cx else -1
        dy = 1 if y < cy else -1
        x += dx
        y += dy

    return len(path)

def crop_border(img_path, threshold=30):
    img = cv2.imread(img_path)
    if img is None:
        return Image.open(img_path)

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray[:,2])
    h, w = gray.shape
    # print(h,w)
    # Distances from 4 corners toward center
    tl = scan_corner_to_center(gray, (1, 1), threshold)
    tr = scan_corner_to_center(gray, (w - 2, 1), threshold)
    bl = scan_corner_to_center(gray, (1, h - 2), threshold)
    br = scan_corner_to_center(gray, (w - 2, h - 2), threshold)

    # print(tl, tr, bl, br)
    # Convert distances to per-side crops (take min from relevant corners)
    crop_left = min(tl, bl)
    crop_right = min(tr, br)
    crop_top = min(tl, tr)
    crop_bottom = min(bl, br)

    # Apply cropping, keeping bounds safe
    x1 = crop_left
    x2 = w - crop_right
    y1 = crop_top
    y2 = h - crop_bottom

    if x2 <= x1 or y2 <= y1:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # fallback

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))




#Visualization

def show_duplicates(df, max_per_group=5, max_to_show = 5):
    dup_hashes = df[df.duplicated('hash', keep=False)]
    grouped = dup_hashes.groupby('hash')
    for hash_val, group in grouped:
        if max_to_show <= 0:
            break
        file_paths = group['file_path'].tolist()[:max_per_group]
        n = len(file_paths)
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        for ax, fp in zip(axes, file_paths):
            ax.imshow(Image.open(fp))
            ax.axis('off')
        plt.suptitle(f"Duplicate group: {hash_val} ({len(group)} images total)")
        plt.show()
        max_to_show -= 1

def show_random_samples(df, label, n=5):
    sample_paths = random.sample(df[df['label'] == label]['file_path'].tolist(), n)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for ax, img_path in zip(axes, sample_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f"Random {label.capitalize()} Samples")
    plt.show()

def show_border_samples(df, n=5):
    subset = df[df['has_vignette_border']].sample(n)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for ax, fp in zip(axes, subset['file_path']):
        ax.imshow(Image.open(fp))
        ax.axis('off')
    plt.suptitle("Sample Images with Vignette Borders")
    plt.show()
    

def show_hair_removal_samples(df):
    import matplotlib.pyplot as plt
    from PIL import Image

    sample_paths = df['file_path'].tolist()
    fig, axes = plt.subplots(4, 2, figsize=(8, 3*4))

    for i, fp in enumerate(sample_paths):
        orig = Image.open(fp).convert("RGB")
        cleaned = remove_hair(orig)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cleaned)
        axes[i, 1].set_title("After Hair Removal")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def show_before_after_crop(df, n=5):
    import matplotlib.pyplot as plt

    if(len(df)>10):
        subset = df[df['has_vignette_border']].sample(n)
    
        fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
        for i, (_, row) in enumerate(subset.iterrows()):
            # Original
            orig = Image.open(row['file_path'])
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            # Cropped
            cropped = crop_border(row['file_path'])
            axes[i, 1].imshow(cropped)
            axes[i, 1].set_title("Cropped")
            axes[i, 1].axis('off')
    else:
        subset = df[df['has_vignette_border']]

        fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
        for i, (_, row) in enumerate(subset.iterrows()):
            # Original
            orig = Image.open(row['file_path'])
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            # Cropped
            cropped = crop_border(row['file_path'])
            axes[i, 1].imshow(cropped)
            axes[i, 1].set_title("Cropped")
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()