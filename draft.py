import os
import cv2
from ultralytics import YOLO

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "best.pt"  # Your trained model for license plate detection
INPUT_FOLDER = "images"  # Folder containing input images
OUTPUT_FOLDER = "plates_output"  # Folder where cropped plates will be saved

# ===============================
# INITIALIZATION
# ===============================
# Load the YOLO model
model = YOLO(MODEL_PATH)

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def crop_license_plates():
    """
    Processes images in INPUT_FOLDER, detects license plates, 
    and saves the cropped plates to OUTPUT_FOLDER.
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    print(f"Processing {len(image_files)} images...")

    count = 0
    for image_name in image_files:
        image_path = os.path.join(INPUT_FOLDER, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping: Could not read {image_name}")
            continue

        # Run detection
        results = model(image, verbose=False)

        all_detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                # Class 0 is 'license-plate'
                if int(cls) == 0:
                    all_detections.append({
                        'box': map(int, box[:4]),
                        'conf': conf
                    })

        if all_detections:
            # Sort by confidence descending and pick the best one
            all_detections.sort(key=lambda x: x['conf'], reverse=True)
            best_detection = all_detections[0]
            
            x1, y1, x2, y2 = best_detection['box']
            plate_crop = image[y1:y2, x1:x2]

            # Generate output filename
            base_name = os.path.splitext(image_name)[0]
            output_name = f"{base_name}_plate.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            # Save the cropped plate
            cv2.imwrite(output_path, plate_crop)
            count += 1
            print(f"Saved (conf={best_detection['conf']:.2f}): {output_name}")

    print(f"\nDone! Processed images and saved {count} plate crops to '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    crop_license_plates()
