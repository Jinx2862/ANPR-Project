import os
#pyre-ignore
import cv2

import re
# pyre-ignore[21]
import pandas as pd
# pyre-ignore[21]
from ultralytics import YOLO
# pyre-ignore[21]
import easyocr

# ===============================
# Load Models
# ===============================
model = YOLO("yolov8n.pt")  # Indian plate model
reader = easyocr.Reader(['en'])

# ===============================
# Indian Plate Validation & Correction
# ===============================
def validate_and_format_plate(text):
    # Basic cleaning
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if not text:
        return None

    # Common character corrections (OCR fixes)
    digit_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B','4':'A','6':'G'}
    char_to_digit = {v: k for k, v in digit_to_char.items()}
    char_to_digit.update({'D': '0', 'Q': '0', 'G': '6', 'T': '7', 'Z': '2'})

    # Handle standard 10-char or 9-char plates
    # AA NN AA NNNN or AA NN A NNNN or AA NN NNNN
    new_text = list(text)
    
    # Position 0, 1: Should be State Code (Letters)
    for i in range(min(2, len(new_text))):
        if new_text[i].isdigit() and new_text[i] in digit_to_char:
            new_text[i] = digit_to_char[new_text[i]]

    # Position 2, 3: Should be District Code (Numbers)
    for i in range(2, min(4, len(new_text))):
        if new_text[i].isalpha() and new_text[i] in char_to_digit:
            new_text[i] = char_to_digit[new_text[i]]

    # The rest is a mix, but often the last 4 are numbers
    if len(new_text) >= 4:
        last_indices = range(max(4, len(new_text) - 4), len(new_text))
        for i in last_indices:
            if new_text[i].isalpha() and new_text[i] in char_to_digit:
                new_text[i] = char_to_digit[new_text[i]]

    return "".join(new_text)

def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# ===============================
# Folders
# ===============================
image_folder = "images"
output_folder = "test_images"

os.makedirs(output_folder, exist_ok=True)

results_list = []

# ===============================
# Process Images
# ===============================
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(image)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box[:4])

            # Crop detected plate
            plate = image[y1:y2, x1:x2]

            # OCR Preprocessing Improvements
            # 1. Resize for better OCR visibility
            height, width = plate.shape[:2]
            scaling_factor = 2.0  # Upscale
            plate_resized = cv2.resize(plate, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_CUBIC)

            # 2. Convert to gray and apply filters
            plate_gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
            plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

            # 3. Adaptive Thresholding
            plate_thresh = cv2.adaptiveThreshold(
                plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            ocr_results = reader.readtext(plate_thresh)

            for (_, text, prob) in ocr_results:
                cleaned = validate_and_format_plate(text)

                if cleaned and len(cleaned) >= 6:
                    print(f"{image_name} → {cleaned}")

                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw plate text
                    cv2.putText(
                        image,
                        cleaned,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    results_list.append({
                        "image_name": image_name,
                        "detected_plate": cleaned,
                        "yolo_confidence": round(float(conf), 2),
                        "ocr_confidence": round(float(prob), 2)
                    })

    # Save annotated image
    cv2.imwrite(os.path.join(output_folder, image_name), image)

# ===============================
# Save CSV
# ===============================
df = pd.DataFrame(results_list)
df.to_csv("op_file.csv", index=False)

print("\n✅ DONE! CSV + Annotated images saved.")