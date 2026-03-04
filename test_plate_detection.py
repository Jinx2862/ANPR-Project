from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source="sample_video.mp4",
    save=True,
    conf=0.4
)
