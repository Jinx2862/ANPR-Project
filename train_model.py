# pyre-ignore[21]
from ultralytics import YOLO
import os

# Path to your YAML file
data_yaml_path = "datasets/license_plate_data/config/data.yaml"

if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"❌ data.yaml not found at {data_yaml_path}")

print(f"Using dataset config: {data_yaml_path}")

# Load YOLOv8 Nano (best for CPU)
model = YOLO("yolo26n.pt")

# Train
model.train(
    data=data_yaml_path,
    epochs=20,
    imgsz=640,
    batch=4,        # safer for CPU
    device="cpu",
    workers=0
)

print("\n✅ Training Complete!")
print("Best model saved at: runs/detect/train/weights/best.pt")