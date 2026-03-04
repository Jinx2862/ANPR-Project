import os
# pyre-ignore[21]
import cv2
import re
# pyre-ignore[21]
import pandas as pd
# pyre-ignore[21]
from ultralytics import YOLO
# pyre-ignore[21]
import easyocr

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "best.pt"  # your trained model
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output_images"
CSV_NAME = "anpr_results.csv"

# ===============================
# LOAD MODELS
# ===============================
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results_list = []

# ===============================
# PLATE CLEANING FUNCTION
# ===============================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# ===============================
# PROCESS IMAGES
# ===============================
for image_name in os.listdir(IMAGE_FOLDER):

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(image)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):

            x1, y1, x2, y2 = map(int, box[:4])

            # Class 0 = license-plate
            # Class 1 = vehicle
            if int(cls) == 0:  # Only process license plates

                plate_crop = image[y1:y2, x1:x2]

                # -----------------------------
                # OCR Preprocessing
                # -----------------------------
                plate_resized = cv2.resize(
                    plate_crop, None, fx=2, fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)

                thresh = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
                )

                ocr_results = reader.readtext(
                    thresh,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                for (_, text, prob) in ocr_results:

                    plate_number = clean_plate(text)

                    if len(plate_number) >= 6:

                        print(f"{image_name} → {plate_number}")

                        # Draw plate bounding box (Green)
                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        # Draw detected plate text
                        cv2.putText(
                            image,
                            plate_number,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        results_list.append({
                            "image_name": image_name,
                            "detected_plate": plate_number,
                            "yolo_confidence": round(float(conf), 2),
                            "ocr_confidence": round(float(prob), 2)
                        })

            elif int(cls) == 1:
                # Draw vehicle bounding box (Blue)
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

    # Save annotated image
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name), image)

# ===============================
# SAVE CSV
# ===============================
df = pd.DataFrame(results_list)
df.to_csv(CSV_NAME, index=False)

print("\n✅ DONE! CSV + Annotated images saved.")