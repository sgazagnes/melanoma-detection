# modelmole.py — full pipeline: camera discovery → capture → inference → display
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import time

# ── CONFIG ───────────────────────────────────────────────────────────────────
CKPT             = "models/baseline_mnv3.pt"
ROI_RADIUS       = 150
SHARP_THRESHOLD  = 100
COUNTDOWN_SEC    = 3
CENTER_TOLERANCE = 0.35
N_FRAMES         = 15
PADDING          = 100
TARGET_SIZE      = 224
THRESHOLD        = 0.5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ────────────────────────────────────────────────────────────────────
ckpt      = torch.load(CKPT, map_location=DEVICE)
mean, std = ckpt["mean"], ckpt["std"]

def build_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m

model = build_model()
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

# ── CAMERA DISCOVERY ─────────────────────────────────────────────────────────
def find_camera():
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Camera found at index {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        cap.release()
    raise RuntimeError("No camera found (tried indices 0-5)")

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_roi(frame, cx, cy, radius):
    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius, cy + radius
    return frame[y1:y2, x1:x2]

def detect_mole(roi):
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    mean_val = blurred.mean()
    _, thresh = cv2.threshold(blurred, mean_val * 0.65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None
    roi_cx, roi_cy = roi.shape[1] // 2, roi.shape[0] // 2
    max_dist = ROI_RADIUS * CENTER_TOLERANCE
    centered = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])
        dist = np.sqrt((cx_ - roi_cx)**2 + (cy_ - roi_cy)**2)
        if dist < max_dist:
            centered.append((c, dist))
    if not centered:
        return False, None
    largest, _ = min(centered, key=lambda x: x[1])
    area        = cv2.contourArea(largest)
    roi_area    = 3.14 * ROI_RADIUS**2 / 5
    perimeter   = cv2.arcLength(largest, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    is_mole = 0.005 < area / roi_area < 0.60 and circularity > 0.2
    return is_mole, largest

def sharpness(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tight_crop(roi_bgr, contour):
    x, y, cw, ch = cv2.boundingRect(contour)
    h, w = roi_bgr.shape[:2]
    x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
    x2, y2 = min(w, x + cw + PADDING), min(h, y + ch + PADDING)
    crop = roi_bgr[y1:y2, x1:x2]
    upscaled = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

def apply_clahe(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

def run_inference(roi_bgr):
    _, contour = detect_mole(roi_bgr)
    mole_pil = tight_crop(roi_bgr, contour) if contour is not None \
               else Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    enhanced = apply_clahe(mole_pil)
    x = tfm(enhanced).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    label = "MALIGNANT" if prob >= THRESHOLD else "BENIGN"
    return prob, label, mole_pil, enhanced

def build_result_panel(roi_bgr, mole_pil, enhanced, prob, label, h):
    PANEL_W  = 400
    LABEL_H  = 30  # height reserved for text under each thumbnail
    MARGIN   = 10
    thumb_h  = (h - 4 * MARGIN - 3 * LABEL_H - 100) // 3  # 100px for result text
    thumb_w  = PANEL_W - 2 * MARGIN
    panel    = np.zeros((h, PANEL_W, 3), dtype=np.uint8)

    def pil_to_bgr_resized(img, target_w, target_h):
        """Resize preserving aspect ratio, pad with black to fill."""
        iw, ih = img.size
        scale  = min(target_w / iw, target_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        canvas  = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    positions = [
        (MARGIN,                              "ROI",        Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))),
        (MARGIN + thumb_h + LABEL_H + MARGIN, "Tight crop", mole_pil),
        (MARGIN + 2*(thumb_h + LABEL_H + MARGIN), "CLAHE",  enhanced),
    ]

    for y, title, img in positions:
        thumb = pil_to_bgr_resized(img, thumb_w, thumb_h)
        panel[y:y+thumb_h, MARGIN:MARGIN+thumb_w] = thumb
        cv2.putText(panel, title, (MARGIN, y + thumb_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Result text at bottom
    color  = (0, 0, 255) if label == "MALIGNANT" else (0, 200, 0)
    text_y = MARGIN + 3 * (thumb_h + LABEL_H + MARGIN)
    cv2.putText(panel, label,                    (MARGIN, text_y + 25),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(panel, f"Conf: {prob:.1%}",      (MARGIN, text_y + 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(panel, "N = new scan  Q = quit", (MARGIN, text_y + 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
cap             = find_camera()
countdown_start = None
burst_frames    = []
result_panel    = None   # None = no result yet
paused_roi      = None   # ROI used for last inference

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    roi = get_roi(clean_frame, cx, cy, ROI_RADIUS)

    # ── Detection & sharpness ─────────────────────────────────────────────
    is_mole, contour = detect_mole(roi)
    sharp_val        = sharpness(roi)
    is_sharp         = sharp_val > SHARP_THRESHOLD
    ready            = is_mole and is_sharp

    # ── Countdown ────────────────────────────────────────────────────────
    if result_panel is None:  # only count down when no result is showing
        if ready:
            if countdown_start is None:
                countdown_start = time.time()
            elapsed   = time.time() - countdown_start
            remaining = max(0, COUNTDOWN_SEC - elapsed)
        else:
            countdown_start = None
            remaining       = COUNTDOWN_SEC
            burst_frames    = []
    else:
        remaining = COUNTDOWN_SEC  # frozen while result shown

    # ── Burst collection ─────────────────────────────────────────────────
    if result_panel is None and ready and remaining == 0:
        burst_frames.append((sharp_val, roi.copy()))

    # ── Inference trigger ─────────────────────────────────────────────────
    if result_panel is None and len(burst_frames) >= N_FRAMES:
        best_roi    = max(burst_frames, key=lambda x: x[0])[1]
        paused_roi  = best_roi
        roi_bgr     = paused_roi
        print("Running inference...")
        prob, label, mole_pil, enhanced = run_inference(roi_bgr)
        result_panel = build_result_panel(roi_bgr, mole_pil, enhanced, prob, label, h)
        print(f"Result: {label}  ({prob:.1%})")
        burst_frames    = []
        countdown_start = None

    # ── Overlays ──────────────────────────────────────────────────────────
    if result_panel is None:
        color = (0, 255, 0) if ready else (0, 0, 255)
        if not is_mole:
            msg = "Center the mole in the circle"
        elif not is_sharp:
            msg = "Hold still / move closer"
        elif remaining > 0:
            msg = f"Hold still... {remaining:.1f}s"
        else:
            msg = f"Capturing... {len(burst_frames)}/{N_FRAMES}"
            color = (0, 200, 255)

        if contour is not None:
            offset_contour = contour + np.array([cx - ROI_RADIUS, cy - ROI_RADIUS])
            cv2.drawContours(frame, [offset_contour], -1, color, 2)

        progress = int(((COUNTDOWN_SEC - remaining) / COUNTDOWN_SEC) * (ROI_RADIUS * 2)) \
                   if remaining > 0 else \
                   int((len(burst_frames) / N_FRAMES) * (ROI_RADIUS * 2))
        cv2.rectangle(frame, (cx - ROI_RADIUS, cy + ROI_RADIUS + 10),
                             (cx - ROI_RADIUS + progress, cy + ROI_RADIUS + 20),
                             (0, 255, 0), -1)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.circle(frame, (cx, cy), ROI_RADIUS, color, 2)
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Sharp: {sharp_val:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        # Keep circle visible while result is shown
        cv2.circle(frame, (cx, cy), ROI_RADIUS, (100, 100, 100), 1)
        cv2.putText(frame, "Press N for new scan  |  Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Compose side-by-side ─────────────────────────────────────────────
    if result_panel is not None:
        panel = result_panel
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (panel.shape[1], h))
        display = np.hstack([frame, panel])
    else:
        display = frame

    cv2.imshow("ModelMole", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        result_panel    = None
        paused_roi      = None
        burst_frames    = []
        countdown_start = None

cap.release()
cv2.destroyAllWindows()