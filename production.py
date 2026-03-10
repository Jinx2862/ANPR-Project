# ============================================
# IMPORT REQUIRED LIBRARIES
# ============================================

import cv2                         # OpenCV for video processing
import time                        # Used for FPS calculation
import pandas as pd                # Used for saving plate data to CSV
import re                          # Regex for plate pattern matching
from ultralytics import YOLO       # YOLO object detection model
import easyocr                     # OCR engine
from deep_sort_realtime.deepsort_tracker import DeepSort  # Tracking


# ============================================
# CONFIGURATION PARAMETERS
# ============================================

MODEL_PATH = "best.pt"                 # Path to trained YOLO model
CSV_FILE = "detected_plates.csv"       # Output CSV file
DETECTION_CONF = 0.30                  # Minimum YOLO confidence
FRAME_SKIP = 3                         # Skip frames for CPU optimization


# ============================================
# LOAD MODELS
# ============================================

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize OCR reader (English characters)
reader = easyocr.Reader(['en'], gpu=False)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)


# ============================================
# INDIAN NUMBER PLATE DATA
# ============================================

# List of valid Indian state codes
STATE_CODES = [
'AN','AP','AR','AS','BR','CH','CG','DD','DL','GA','GJ','HR','HP',
'JK','JH','KA','KL','LA','LD','MP','MH','ML','MZ','NL','OR','OD',
'PY','PB','RJ','SK','TN','TS','TR','UP','UK','WB'
]


# ============================================
# MEMORY CACHE + VEHICLE COUNT
# ============================================

# Store already detected plates
plate_memory = set()

# Vehicle counter
vehicle_count = 0


# ============================================
# CREATE CSV FILE
# ============================================

# Create CSV with headers
df = pd.DataFrame(columns=["timestamp", "plate_number", "track_id"])
df.to_csv(CSV_FILE, index=False)


# ============================================
# TEXT CLEANING FUNCTION
# ============================================

def clean_plate_text(text):
    """
    Removes unwanted characters from OCR output
    Keeps only letters and numbers.
    """

    text = text.upper()  # Convert to uppercase

    # Remove everything except A-Z and 0-9
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


# ============================================
# INDIAN PLATE CORRECTION LOGIC
# ============================================

def correct_indian_plate(text):
    """
    Detects and formats Indian plates.
    Supports:
    1) State series (MH12AB1234)
    2) BH series (22BH1234AA)
    """

    if not text or len(text) < 7:
        return ""

    text = text.upper()


    # ---------------------------------
    # BH SERIES LOGIC
    # ---------------------------------
    bh_pattern = re.compile(r'([0-9]{2})BH([0-9]{4})([A-Z]{2})')

    match = bh_pattern.search(text)

    if match:
        year, number, series = match.groups()

        return f"{year}BH{number}{series}"


    # ---------------------------------
    # STATE SERIES LOGIC
    # ---------------------------------
    state_pattern = re.compile(
        r'([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{3,4})'
    )

    match = state_pattern.search(text)

    if match:

        state, district, series, number = match.groups()

        # Validate state code
        if state in STATE_CODES:

            district = district.zfill(2)

            return f"{state}{district}{series}{number}"

    return ""


# ============================================
# OCR FUNCTION
# ============================================

def read_plate(plate_crop):
    """
    Performs OCR and applies Indian plate correction logic.
    """

    # Run OCR
    results = reader.readtext(plate_crop)

    raw_text = ""

    # Combine detected OCR text
    for res in results:
        raw_text += res[1]

    # Step 1: Clean OCR output
    cleaned_text = clean_plate_text(raw_text)

    # Step 2: Apply Indian correction logic
    corrected_text = correct_indian_plate(cleaned_text)

    return corrected_text


# ============================================
# VIDEO SOURCE
# ============================================

# Use webcam
cap = cv2.VideoCapture(0)

# For video file use:
# cap = cv2.VideoCapture("traffic.mp4")


frame_count = 0

start_time = time.time()


# ============================================
# MAIN LOOP
# ============================================

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames to increase speed
    if frame_count % FRAME_SKIP != 0:
        continue


    # ======================================
    # YOLO PLATE DETECTION
    # ======================================

    results = model(frame, conf=DETECTION_CONF, verbose=False)

    detections = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            if int(cls) == 0:  # license plate class

                x1, y1, x2, y2 = map(int, box)

                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'plate'))


    # ======================================
    # TRACKING USING DEEPSORT
    # ======================================

    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, w, h = map(int, track.to_ltrb())

        plate_crop = frame[t:t+h, l:l+w]


        # ======================================
        # OCR + INDIAN PLATE LOGIC
        # ======================================

        plate_text = read_plate(plate_crop)


        # ======================================
        # DUPLICATE PLATE FILTER
        # ======================================

        if plate_text and plate_text not in plate_memory:

            plate_memory.add(plate_text)

            vehicle_count += 1

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            new_row = pd.DataFrame(
                [[timestamp, plate_text, track_id]],
                columns=["timestamp", "plate_number", "track_id"]
            )

            new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)

            print("Detected:", plate_text)


        # ======================================
        # DRAW RESULTS
        # ======================================

        cv2.rectangle(frame, (l, t), (l+w, t+h), (0,255,0), 2)

        cv2.putText(
            frame,
            plate_text,
            (l, t-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    # ======================================
    # VEHICLE COUNT DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        3
    )


    # ======================================
    # FPS DISPLAY
    # ======================================

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


    # Show video
    cv2.imshow("Advanced Real-Time ANPR", frame)


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()