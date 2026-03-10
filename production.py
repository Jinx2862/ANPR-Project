# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import cv2                     # OpenCV for video capture and drawing
import time                    # Used for FPS calculation
import pandas as pd            # Used for logging plates to CSV
import re                      # Used for plate pattern matching
from collections import Counter  # Used for majority voting
from ultralytics import YOLO   # YOLO detection model
import easyocr                 # OCR engine
from deep_sort_realtime.deepsort_tracker import DeepSort  # Tracking


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "best.pt"             # YOLO model path
CSV_FILE = "detected_plates.csv"   # CSV output file
DETECTION_CONF = 0.30              # YOLO confidence threshold
FRAME_SKIP = 3                     # Skip frames for speed
OCR_BUFFER_SIZE = 5                # Frames used for majority voting


# ==========================================================
# LOAD MODELS
# ==========================================================

model = YOLO(MODEL_PATH)           # Load YOLO model

reader = easyocr.Reader(['en'], gpu=False)  # Initialize OCR

tracker = DeepSort(max_age=30)     # Initialize DeepSORT tracker


# ==========================================================
# INDIAN STATE CODES
# ==========================================================

STATE_CODES = [
'AN','AP','AR','AS','BR','CH','CG','DD','DL','GA','GJ','HR','HP',
'JK','JH','KA','KL','LA','LD','MP','MH','ML','MZ','NL','OR','OD',
'PY','PB','RJ','SK','TN','TS','TR','UP','UK','WB'
]


# ==========================================================
# MEMORY STRUCTURES
# ==========================================================

plate_memory = set()        # Stores plates already logged
vehicle_count = 0           # Counts vehicles

ocr_buffers = {}            # Stores OCR results per track ID


# ==========================================================
# CREATE CSV FILE
# ==========================================================

df = pd.DataFrame(columns=["timestamp", "plate_number", "track_id"])
df.to_csv(CSV_FILE, index=False)


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_plate_text(text):

    text = text.upper()

    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


# ==========================================================
# INDIAN PLATE FORMAT CORRECTION
# ==========================================================

def correct_indian_plate(text):

    if not text or len(text) < 7:
        return ""

    text = text.upper()

    # -------------------------
    # BH SERIES FORMAT
    # -------------------------
    bh_pattern = re.compile(r'([0-9]{2})BH([0-9]{4})([A-Z]{2})')

    match = bh_pattern.search(text)

    if match:
        year, number, series = match.groups()
        return f"{year}BH{number}{series}"


    # -------------------------
    # STATE SERIES FORMAT
    # -------------------------
    state_pattern = re.compile(
        r'([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{3,4})'
    )

    match = state_pattern.search(text)

    if match:

        state, district, series, number = match.groups()

        if state in STATE_CODES:

            district = district.zfill(2)

            return f"{state}{district}{series}{number}"

    return ""


# ==========================================================
# OCR FUNCTION
# ==========================================================

def read_plate(plate_crop):

    results = reader.readtext(plate_crop)

    raw_text = ""

    for res in results:
        raw_text += res[1]

    cleaned = clean_plate_text(raw_text)

    corrected = correct_indian_plate(cleaned)

    return corrected


# ==========================================================
# MAJORITY VOTING FUNCTION
# ==========================================================

def majority_vote(text_list):
    """
    Returns most frequent plate from OCR results
    """

    if len(text_list) == 0:
        return ""

    counter = Counter(text_list)

    return counter.most_common(1)[0][0]


# ==========================================================
# VIDEO SOURCE
# ==========================================================

cap = cv2.VideoCapture(0)

frame_count = 0

start_time = time.time()


# ==========================================================
# MAIN PROCESSING LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames for performance
    if frame_count % FRAME_SKIP != 0:
        continue


    # ======================================================
    # YOLO LICENSE PLATE DETECTION
    # ======================================================

    results = model(frame, conf=DETECTION_CONF, verbose=False)

    detections = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            if int(cls) == 0:

                x1, y1, x2, y2 = map(int, box)

                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'plate'))


    # ======================================================
    # TRACKING USING DEEPSORT
    # ======================================================

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, w, h = map(int, track.to_ltrb())

        plate_crop = frame[t:t+h, l:l+w]


        # ======================================================
        # OCR EXTRACTION
        # ======================================================

        plate_text = read_plate(plate_crop)

        if plate_text == "":
            continue


        # ======================================================
        # OCR MAJORITY VOTING BUFFER
        # ======================================================

        if track_id not in ocr_buffers:

            ocr_buffers[track_id] = []

        ocr_buffers[track_id].append(plate_text)

        # Limit buffer size
        if len(ocr_buffers[track_id]) > OCR_BUFFER_SIZE:

            ocr_buffers[track_id].pop(0)


        # Get most frequent OCR result
        final_plate = majority_vote(ocr_buffers[track_id])


        # ======================================================
        # DUPLICATE FILTER
        # ======================================================

        if final_plate and final_plate not in plate_memory:

            plate_memory.add(final_plate)

            vehicle_count += 1

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            new_row = pd.DataFrame(
                [[timestamp, final_plate, track_id]],
                columns=["timestamp","plate_number","track_id"]
            )

            new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)

            print("Detected Plate:", final_plate)


        # ======================================================
        # DRAW BOUNDING BOX
        # ======================================================

        cv2.rectangle(frame,(l,t),(l+w,t+h),(0,255,0),2)

        cv2.putText(
            frame,
            final_plate,
            (l,t-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    # ======================================================
    # DISPLAY VEHICLE COUNT
    # ======================================================

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        3
    )


    # ======================================================
    # FPS DISPLAY
    # ======================================================

    elapsed_time = time.time() - start_time

    fps = frame_count / elapsed_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        3
    )


    # ======================================================
    # SHOW VIDEO WINDOW
    # ======================================================

    cv2.imshow("Advanced ANPR System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()