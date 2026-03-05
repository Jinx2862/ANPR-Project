# pyre-ignore-all-errors
# pyre-ignore[21]
import cv2
# pyre-ignore[21]
import easyocr
# pyre-ignore[21]
import numpy as np
import re
from collections import Counter
# pyre-ignore[21]
from ultralytics import YOLO

# ===============================
# Load Models
# ===============================
model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture("sample_video.mp4")

frame_count = 0
ocr_interval = 5
plate_buffer = []
stable_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    results = model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Run OCR every N frames
            if frame_count % ocr_interval == 0:

                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                    continue

                # Upscale
                plate_crop = cv2.resize(
                    plate_crop,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                ocr_results = reader.readtext(gray)

                for (_, text, confidence) in ocr_results:
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    print("RAW:", cleaned, "CONF:", confidence)

                    if confidence > 0.15 and len(cleaned) >= 4:
                        plate_buffer.append(cleaned)
                        print("Raw:", cleaned)

            # ==========================
            # Majority Voting
            # ==========================
            if len(plate_buffer) >= 10:
                most_common = Counter(plate_buffer).most_common(1)[0][0]
                stable_text = most_common
                plate_buffer.clear()
                print("STABLE PLATE:", stable_text)

            if stable_text != "":
                cv2.putText(frame, stable_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,255,0), 2)

    frame_count += 1

    cv2.imshow("Stable ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
