import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def debug_detection(image_name):
    image_path = os.path.join("images", image_name)
    if not os.path.exists(image_path):
        print(f"File {image_path} not found.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_name}")
        return

    print(f"\n--- Analyzing {image_name} ---")
    # Run prediction with very low confidence to see everything
    results = model(image, verbose=False, conf=0.01) 
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            print("No detections found even at 0.01 confidence.")
        else:
            for box, cls, conf in zip(boxes, classes, confidences):
                cls_name = model.names[int(cls)]
                print(f"Detected: {cls_name} (Class {int(cls)}) | Confidence: {conf:.4f} | Box: {box}")

    # Try with larger imgsz
    print(f"\n--- Analyzing {image_name} (imgsz=1280) ---")
    results_hd = model(image, verbose=False, conf=0.05, imgsz=1280)
    for result in results_hd:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            print("No detections found at imgsz=1280.")
        else:
            for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                cls_name = model.names[int(cls)]
                print(f"Detected: {cls_name} (Class {int(cls)}) | Confidence: {conf:.4f} | Box: {box}")

if __name__ == "__main__":
    debug_detection("1.jpg")
    debug_detection("2.jpg")
