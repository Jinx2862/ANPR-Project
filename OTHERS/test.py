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
# Load YOLO Model (CPU Mode)
# ===============================
model = YOLO("best.pt")
model.to("cpu")  # Force CPU

reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# Indian Plate Regex
# ===============================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Indian format: MH12AB1234
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$'
    if re.match(pattern, text):
        return text

    return text

# ===============================
# Folders
# ===============================
image_folder = "images"
output_folder = "images_output"
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

    results = model(image, conf=0.4)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confidences):

            x1, y1, x2, y2 = map(int, box[:4])
            plate = image[y1:y2, x1:x2]

            # Preprocessing for better OCR
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
            plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)[1]

            # OCR
            ocr_results = reader.readtext(plate_thresh)

            for (_, text, prob) in ocr_results:

                cleaned = clean_plate(text)

                if len(cleaned) >= 6:

                    print(f"{image_name} → {cleaned}")

                    # Draw Bounding Box
                    cv2.rectangle(
                        image, # image
                        (x1, y1), # starting point
                        (x2, y2), # ending point
                        (0, 255, 0), # color
                        2 # thickness
                    )

                    # Draw Plate Text
                    cv2.putText(
                        image, # image
                        cleaned, # text
                        (x1, y1 - 10), # position
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        0.8, # size
                        (0, 255, 0), # color
                        2 # thickness
                    )

                    results_list.append({
                        "image_name": image_name, # image name
                        "detected_plate": cleaned, # detected plate
                        "yolo_confidence": round(float(conf), 2), # yolo confidence
                        "ocr_confidence": round(float(prob), 2) # ocr confidence
                    })

    # Save annotated image
    cv2.imwrite(os.path.join(output_folder, image_name), image)

# ===============================
# Save CSV
# ===============================
df = pd.DataFrame(results_list)
df.to_csv("ouput_file.csv", index=False)

print("\n✅ DONE! Check output_images and output.csv")