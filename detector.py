# ============================================================
# PILL & CAPSULE DETECTOR
# YOLOv8 trained model — INFERENCE ONLY.
#
# This module does NOT open a camera and does NOT show a window.
# The browser owns the webcam; app.py hands this module one
# decoded frame at a time (a numpy/OpenCV BGR image) and gets
# back structured detection data.
# ============================================================

import time
from collections import Counter

import torch
from ultralytics import YOLO


# ------------------------------------------------------------
# YOUR TRAINED MODEL
# Change only this line if you move the model file.
# ------------------------------------------------------------

MODEL_PATH = r"C:\Users\Acer\Downloads\object counting\model.pt"

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONFIDENCE = 0.45          # Confidence thoda barha diya hai taaki ghalat cheezein filter ho jayein
IMAGE_SIZE = 500           # larger can help with small pills/capsules
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# LOAD MODEL (once, when this module is imported by app.py)
# ------------------------------------------------------------

print("=" * 60)
print("Loading trained YOLO model...")
print("=" * 60)
print("Model:", MODEL_PATH)

try:
    model = YOLO(MODEL_PATH)

except FileNotFoundError:
    print("\nERROR: Model file not found.")
    print("Check this path:")
    print(MODEL_PATH)
    raise

except Exception as e:
    print("\nERROR loading model:")
    print(e)
    print("\nMake sure Ultralytics/PyTorch are installed correctly.")
    raise

print("\nModel loaded successfully.")
print("Classes:", model.names)
print("Device:", "GPU (cuda:0)" if DEVICE == 0 else "CPU")
print("=" * 60)


# ------------------------------------------------------------
# DETECT OBJECTS IN A SINGLE FRAME
# ------------------------------------------------------------

def detect_objects(frame, confidence: float = None):
    """
    Run YOLO inference on one OpenCV BGR frame (numpy array).

    Called by app.py's /detect endpoint for every frame the
    browser uploads. Never opens a camera, never shows a window.

    Args:
        frame: numpy ndarray (BGR image), e.g. from cv2.imdecode.
        confidence: optional override for the confidence threshold.

    Returns:
        {
            "total": int,
            "counts": {class_name: count, ...},
            "detections": [
                {
                    "class_name": str,
                    "confidence": float,
                     "box": {"x1": float, "y1": float, "x2": float, "y2": float}
                },
                ...
            ],
            "fps": float,
            "confidence": float   # threshold actually used
        }
    """

    conf = confidence if confidence is not None else CONFIDENCE

    start_time = time.time()

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]

    counts = Counter()
    total = 0
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        for i in range(len(result.boxes)):

            cls_id = int(result.boxes.cls[i].item())
            box_confidence = float(result.boxes.conf[i].item())

            # Get class name straight from the trained model
            if isinstance(model.names, dict):
                class_name = model.names.get(cls_id, str(cls_id))
            else:
                class_name = model.names[cls_id]

            box = result.boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            # --- JUGAAR / SIZE FILTER ---
            # Tablets aur capsules chote hote hain. Agar koi box bohot bada hai 
            # (jaise poori tasveer ya background object), toh usko ignore kar dein.
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Agar area ek limit (misal ke taur par 20000 pixels) se zyada hai, toh skip karein
            MAX_ALLOWABLE_AREA = 20000 
            if box_area > MAX_ALLOWABLE_AREA:
                continue  # Yeh tablet/capsule nahi hai, background ki badi cheez hai
            # -----------------------------

            counts[class_name] += 1
            total += 1

            detections.append({
                "class_name": class_name,
                "confidence": box_confidence,
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            })

    elapsed = time.time() - start_time
    fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

    return {
        "total": total,
        "counts": dict(counts),
        "detections": detections,
        "fps": fps,
        "confidence": conf,
    }