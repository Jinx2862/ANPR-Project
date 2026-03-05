import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import Counter

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "best.pt"
INPUT_VIDEO = "sample_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
DETECTION_CONF = 0.15 # Slightly higher for video to reduce jitter
HISTORY_WINDOW = 15   # Number of frames to look back for OCR stabilization

# ===============================
# INITIALIZATION
# ===============================
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ==================================
# INDIAN OCR CORRECTION LOGIC
# (Reused from final_draft.py)
# ==================================

LETTER_CORRECTIONS = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A', '7': 'T'
}

DIGIT_CORRECTIONS = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'A': '4',
    'D': '0', 'Q': '0', 'U': '0'
}

STATE_CODES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 
    'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'ML', 'MZ', 'NL', 'OR', 'OD', 
    'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
]

def correct_indian_plate(text):
    if not text or len(text) < 7:
        return text

    normalized = ""
    for char in text:
        if char in ['0', 'O', 'D', 'Q']: normalized += '0'
        elif char in ['1', 'I']: normalized += '1'
        elif char in ['8', 'B']: normalized += '8'
        elif char in ['5', 'S']: normalized += '5'
        elif char in ['2', 'Z']: normalized += '2'
        else: normalized += char

    bh_regex = re.compile(r'([0-9]{1,2})([B8][H])([0-9]{1,4})([A-Z0-9]{1,2})')
    match_bh = bh_regex.search(normalized)
    
    if match_bh:
        y, bh, n, s = match_bh.groups()
        year = "".join([DIGIT_CORRECTIONS.get(c, c) for c in text[match_bh.start(1):match_bh.end(1)]])
        number = "".join([DIGIT_CORRECTIONS.get(c, c) for c in text[match_bh.start(3):match_bh.end(3)]])
        series = "".join([LETTER_CORRECTIONS.get(c, c) for c in text[match_bh.start(4):match_bh.end(4)]])
        return f"{year}BH{number}{series}"

    state_regex = re.compile(r'([A-Z0-9]{2})([0-9]{1,2})([A-Z0-9]{1,2})([0-9]{1,4})')
    match_state = state_regex.search(normalized)
    
    if match_state:
        st, dist, ser, num = match_state.groups()
        orig_st = text[match_state.start(1):match_state.end(1)]
        orig_dist = text[match_state.start(2):match_state.end(2)]
        orig_ser = text[match_state.start(3):match_state.end(3)]
        orig_num = text[match_state.start(4):match_state.end(4)]

        state = "".join([LETTER_CORRECTIONS.get(c, c) for c in orig_st])
        mapping = {
            'HH': 'MH', 'MK': 'MH', 'MM': 'MH', 'NH': 'MH', 'MR': 'MH', 'MI': 'MH', 'HK': 'MH', 'HZ': 'MH', 'M3': 'MH', 'M8': 'MH',
            'KR': 'HR', 'HA': 'HR', 'H2': 'HR', 'K8': 'HR', 'N2': 'HR', 'M2': 'HR',
            'KA': 'KA', 'K6': 'KA', 'K4': 'KA', 'KI': 'KA', 'K1': 'KA',
            'DL': 'DL', 'DI': 'DL', 'D1': 'DL', '0L': 'DL', 'OL': 'DL', 'LL': 'DL'
        }
        state = mapping.get(state, state)
        if len(state) == 2 and state not in STATE_CODES:
            if state[0] == 'M': state = 'MH'
            elif state[0] == 'H': state = 'HR'

        district = "".join([DIGIT_CORRECTIONS.get(c, c) for c in orig_dist])
        if len(district) == 1: district = "0" + district
        series = "".join([LETTER_CORRECTIONS.get(c, c) for c in orig_ser])
        number = "".join([DIGIT_CORRECTIONS.get(c, c) for c in orig_num])
        
        return f"{state}{district}{series}{number}"

    return text

def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def get_best_ocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing scenarios
    resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    scenarios = [gray, resized, thresh] # Reduced scenarios for speed in video
    best_text = ""
    max_score = -1

    for img in scenarios:
        results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not results: continue
        
        raw_text = " ".join([res[1] for res in results])
        confidence = sum([res[2] for res in results]) / len(results)
        
        cleaned = clean_plate_text(raw_text)
        corrected = correct_indian_plate(cleaned)
        
        if not corrected: continue

        score = confidence
        if len(corrected) >= 9: score += 2.0 
        if any(corrected.startswith(sc) for sc in STATE_CODES) or 'BH' in corrected:
            score += 5.0

        if score > max_score:
            max_score = score
            best_text = corrected
        
        if len(corrected) >= 9 and confidence > 0.8 and any(corrected.startswith(sc) for sc in STATE_CODES):
            return corrected

    return best_text

def process_video():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video '{INPUT_VIDEO}' not found.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing video: {INPUT_VIDEO} ({total_frames} frames)")
    
    # Tracking and Stabilization logic
    plate_history = {} # track_id -> list of OCR results
    last_known_plate = {} # track_id -> best OCR result

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. Detect and Track License Plates
        # persist=True enables tracking across frames
        results = model.track(frame, persist=True, verbose=False, conf=DETECTION_CONF)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) == 0: # License plate class
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # 2. Crop and OCR
                    pad = 5
                    py1, py2 = max(0, y1 - pad), min(height, y2 + pad)
                    px1, px2 = max(0, x1 - pad), min(width, x2 + pad)
                    plate_crop = frame[py1:py2, px1:px2]
                    
                    extracted_text = get_best_ocr(plate_crop)
                    
                    # 3. Stabilization Logic
                    if extracted_text:
                        if track_id not in plate_history:
                            plate_history[track_id] = []
                        
                        # Add to history if text is reasonably valid
                        if len(extracted_text) >= 7:
                            plate_history[track_id].append(extracted_text)
                        
                        # Maintain window size
                        if len(plate_history[track_id]) > HISTORY_WINDOW:
                            plate_history[track_id].pop(0)

                    # Determine best text to display
                    if track_id in plate_history and plate_history[track_id]:
                        # Majority vote for text stabilization
                        counts = Counter(plate_history[track_id])
                        best_text = counts.most_common(1)[0][0]
                        last_known_plate[track_id] = best_text
                    else:
                        best_text = last_known_plate.get(track_id, "")

                    # 4. Annotate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if best_text:
                        # Draw label background for better visibility
                        (tw, th), _ = cv2.getTextSize(best_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(frame, best_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Write frame to output
        out.write(frame)
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end='\r')

    print(f"\nProcessing complete. Output saved to: {OUTPUT_VIDEO}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Automatically open the video after processing (Windows specific)
    print(f"Opening {OUTPUT_VIDEO}...")
    os.startfile(OUTPUT_VIDEO)

if __name__ == "__main__":
    process_video()
