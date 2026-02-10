# Biblioteci pentru procesare imagini și segmentare
import os  # Gestionare căi și directoare
import cv2  # OpenCV: citire, procesare și salvare imagini
import numpy as np  # Operații pe matrici (mascări, transformări)
import mediapipe as mp  # Framework Google pentru segmentare persoane

# ===================================
# SECȚIUNEA 1: CONFIGURĂRI - CĂILE DE ACCES
# ===================================
# Detectează automat locația rădăcinii proiectului
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_PATH = os.path.abspath(os.path.join(current_dir, "../.."))

# Căi pentru date brute și procesate
INPUT_DIR  = os.path.join(BASE_PROJECT_PATH, "data", "raw", "synthetic_faces")  # Imagini originale
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, "data", "processed", "Om")  # Imagini procesate (clasa Om)

TARGET_SIZE = (64, 64)  # Resize la 64x64 (compatibil cu modelul CNN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# SECȚIUNEA 2: INIȚIALIZARE MEDIAPIPE
# ===================================
# Modelul de segmentare MediaPipe detectează persoana din fundal
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)  # model_selection=1: mai exact, mai lent

def remove_background_to_white(img):
   
    # Convertire BGR (cv2) -> RGB (MediaPipe)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Rulează segmentare: vrea să vadă cine e persoană (foreground)
    result = segmenter.process(rgb)
    if result.segmentation_mask is None:
        return img  # Dacă nu detectează nimic, return original
    
    # Mască binară: 1 = persoană, 0 = fundal
    mask = result.segmentation_mask
    condition = mask > 0.5  # Pragul de confidență 50%
    # Creează fundal alb (255 = alb în RGB)
    white_background = np.full(img.shape, 255, dtype=np.uint8)
    # Aplică mască: mențin persoană (True), înlocuiesc fundal cu alb (False)
    output_img = np.where(condition[..., None], img, white_background)
    return output_img

def run_preprocessing():
    """Pipeline complet: validează input -> procesează -> salvează cu 64x64"""
    # Validare: verifică dacă folderul INPUT există
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Nu găsesc folderul: {INPUT_DIR}")
        return

    # Filtrare: doar imagini "Om_*.jpg" (clasa Om din dataset generat)
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith("Om_") and f.lower().endswith(".jpg")]
    print(f"[INFO] Am găsit {len(files)} fișiere de tip 'Om_'.")

    # Procesare: pe fiecare imagine
    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)  # Citire imagine
        
        if img is None:  # Skip dacă citire eșuează
            continue

        # TRANSFORMĂRI: eliminare fundal + resize pt CNN
        img_no_bg = remove_background_to_white(img)  # Segmentare MediaPipe
        img_resized = cv2.resize(img_no_bg, TARGET_SIZE, interpolation=cv2.INTER_AREA)  # Resize la 64x64

        # SALVARE: imagini procesate în OUTPUT_DIR
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, img_resized)
        print(f"[OK] Procesat: {filename}")

    print(f"\n[FINISHED] Imagini procesate în: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_preprocessing()