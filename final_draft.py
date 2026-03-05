import os
import cv2
import re
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "best.pt"
INPUT_FOLDER = "new_images"
OUTPUT_FOLDER = "final_output_Images"
CSV_NAME = "final_results.csv"
DETECTION_CONF = 0.10 # Lowered further for recalcitrant plates

# ===============================
# INITIALIZATION
# ===============================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ==================================
# INDIAN OCR CORRECTION LOGIC
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
    """
    Highly robust correction for Indian plates.
    Supports State Series and BH Series with intelligent pattern matching.
    """
    if not text or len(text) < 7:
        return text

    # Normalization: swapping common OCR mistakes to help pattern matching
    # We create a "matchable" version where all potential letters are letters 
    # and potential digits are digits based on common swaps.
    normalized = ""
    for char in text:
        if char in ['0', 'O', 'D', 'Q']: normalized += '0' # Use 0 as canonical for both O and 0
        elif char in ['1', 'I']: normalized += '1'
        elif char in ['8', 'B']: normalized += '8'
        elif char in ['5', 'S']: normalized += '5'
        elif char in ['2', 'Z']: normalized += '2'
        else: normalized += char

    # Regex search for BH Series (Digits-Letters-Digits-Letters)
    # Allows flexible lengths for noisy OCR
    bh_regex = re.compile(r'([0-9]{1,2})([B8][H])([0-9]{1,4})([A-Z0-9]{1,2})')
    match_bh = bh_regex.search(normalized)
    
    if match_bh:
        y, bh, n, s = match_bh.groups()
        # Source characters from original text to maintain some fidelity before final fix
        year = "".join([DIGIT_CORRECTIONS.get(c, c) for c in text[match_bh.start(1):match_bh.end(1)]])
        number = "".join([DIGIT_CORRECTIONS.get(c, c) for c in text[match_bh.start(3):match_bh.end(3)]])
        series = "".join([LETTER_CORRECTIONS.get(c, c) for c in text[match_bh.start(4):match_bh.end(4)]])
        return f"{year}BH{number}{series}"

    # Regex search for State Series (Letters-Digits-Letters-Digits)
    state_regex = re.compile(r'([A-Z0-9]{2})([0-9]{1,2})([A-Z0-9]{1,2})([0-9]{1,4})')
    match_state = state_regex.search(normalized)
    
    if match_state:
        st, dist, ser, num = match_state.groups()
        orig_st = text[match_state.start(1):match_state.end(1)]
        orig_dist = text[match_state.start(2):match_state.end(2)]
        orig_ser = text[match_state.start(3):match_state.end(3)]
        orig_num = text[match_state.start(4):match_state.end(4)]

        # 1. State Code
        state = "".join([LETTER_CORRECTIONS.get(c, c) for c in orig_st])
        mapping = {
            'HH': 'MH', 'MK': 'MH', 'MM': 'MH', 'NH': 'MH', 'MR': 'MH', 'MI': 'MH', 'HK': 'MH', 'HZ': 'MH', 'M3': 'MH', 'M8': 'MH',
            'KR': 'HR', 'HA': 'HR', 'H2': 'HR', 'K8': 'HR', 'N2': 'HR', 'M2': 'HR',
            'KA': 'KA', 'K6': 'KA', 'K4': 'KA', 'KI': 'KA', 'K1': 'KA',
            'DL': 'DL', 'DI': 'DL', 'D1': 'DL', '0L': 'DL', 'OL': 'DL', 'LL': 'DL'
        }
        state = mapping.get(state, state)
        if len(state) == 2 and state not in STATE_CODES:
            # Final fallback: if first is M or H, it's likely MH or HR
            if state[0] == 'M': state = 'MH'
            elif state[0] == 'H': state = 'HR'

        # 2. District
        district = "".join([DIGIT_CORRECTIONS.get(c, c) for c in orig_dist])
        if len(district) == 1: district = "0" + district
        
        # 3. Series
        series = "".join([LETTER_CORRECTIONS.get(c, c) for c in orig_ser])
        
        # 4. Number
        number = "".join([DIGIT_CORRECTIONS.get(c, c) for c in orig_num])
        
        return f"{state}{district}{series}{number}"

    # If no pattern matches, return original text cleaned
    return text

def get_best_ocr(plate_crop):
    """Try multiple preprocessing steps to get the best OCR result."""
    if plate_crop is None or plate_crop.size == 0:
        return ""

    attempts = []
    
    # 1. Basic Grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    attempts.append(("basic", gray))
    
    # 2. Resized (2x) + Adaptive Threshold
    # 2. Resized + Threshold
    resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    # 4. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(cl1, -1, kernel)
    # 5. Inverse
    inv = cv2.bitwise_not(thresh)

    scenarios = [gray, resized, thresh, cl1, sharp, inv]
    best_text = ""
    max_score = -1

    for img in scenarios:
        results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        raw_text = " ".join([res[1] for res in results])
        confidence = sum([res[2] for res in results]) / len(results) if results else 0
        
        cleaned = clean_plate_text(raw_text)
        corrected = correct_indian_plate(cleaned)
        
        if not corrected: continue

        # Scoring: favor 10-char results and higher confidence
        score = confidence
        if len(corrected) >= 9: score += 2.0 
        
        # Huge bonus for containing a valid state code or BH
        if any(corrected.startswith(sc) for sc in STATE_CODES) or 'BH' in corrected:
            score += 5.0

        if score > max_score:
            max_score = score
            best_text = corrected
        
        # Early exit if we have a high-confidence valid result
        if len(corrected) >= 9 and confidence > 0.8 and any(corrected.startswith(sc) for sc in STATE_CODES):
            return corrected

    return best_text

def clean_plate_text(text):
    """Clean OCR output to maintain only alphanumeric characters."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def process_anpr_pipeline():
    """
    Full pipeline: 
    1. Detect plate using YOLO.
    2. pick best detection.
    3. Crop and extract text using OCR.
    4. Annotate original image.
    5. Save results to CSV.
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    results_data = []
    print(f"Processing {len(image_files)} images...")

    for image_name in image_files:
        image_path = os.path.join(INPUT_FOLDER, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # 1. Detect License Plates
        results = model(image, verbose=False, conf=DETECTION_CONF)
        
        all_detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 0: # License plate class
                    all_detections.append({'box': list(map(int, box[:4])), 'conf': conf})

        # RESCUE: If no plate detected, try higher resolution detection
        if not all_detections:
            # We use imgsz=1280 to help YOLO see smaller/finer details
            # and lower conf slightly for the rescue pass
            results_s = model(image, verbose=False, conf=max(0.05, DETECTION_CONF - 0.05), imgsz=1280)
            for result in results_s:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, cls, conf in zip(boxes, classes, confidences):
                    if int(cls) == 0:
                        all_detections.append({'box': list(map(int, box[:4])), 'conf': conf})

        # 2. Pick the Best Detection or Fallback to Full Image
        use_full_image = False
        if all_detections:
            all_detections.sort(key=lambda x: x['conf'], reverse=True)
            best_detection = all_detections[0]
            if best_detection['conf'] < 0.1: # Very low confidence, maybe full image is better
                use_full_image = True
        else:
            use_full_image = True

        if use_full_image:
            # Fallback A: Assume the whole image is the plate
            extracted_text = get_best_ocr(image)
            x1, y1, x2, y2 = 5, 5, image.shape[1]-5, image.shape[0]-5
            conf_val = 0.05
        else:
            best_detection = all_detections[0]
            x1, y1, x2, y2 = best_detection['box']
            conf_val = best_detection['conf']

            # 3. Crop and OCR
            pad = 5
            h, w, _ = image.shape
            py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
            px1, px2 = max(0, x1 - pad), min(w, x2 + pad)
            plate_crop = image[py1:py2, px1:px2]
            
            extracted_text = get_best_ocr(plate_crop)
            
            # Fallback B: If crop OCR failed, try the whole image anyway
            if not extracted_text:
                extracted_text = get_best_ocr(image)
                # If we find it now, keep the YOLO box for annotation but use the new text

        # 4. Annotate Original Image
        # Draw Box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw Text
        if extracted_text:
            cv2.putText(image, extracted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 5. Save Results
        output_path = os.path.join(OUTPUT_FOLDER, image_name)
        cv2.imwrite(output_path, image)

        results_data.append({
            "image_name": image_name,
            "plate_number": extracted_text,
            "confidence": round(float(conf_val), 2)
        })
        print(f"Processed {image_name}: {extracted_text if extracted_text else 'No text extracted'}")

    # 6. Save to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv(CSV_NAME, index=False)
        print(f"\nDone! Results saved to {CSV_NAME} and annotated images saved to '{OUTPUT_FOLDER}'.")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    process_anpr_pipeline()
