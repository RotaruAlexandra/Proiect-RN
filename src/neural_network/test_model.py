import torch
import cv2
import csv
import os
import glob
import numpy as np
import time
from datetime import datetime
# Importăm clasa modelului din fișierul tău de definiție
from model_torch import FaceClassifierCNN

# =========================================================
# 1. IDLE (Configurări inițiale)
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_PATH = os.path.abspath(os.path.join(current_dir, "../.."))

# Căile către resurse conform structurii tale
MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "models/best_CNN_Principal.pt")
LOG_CSV = os.path.join(BASE_PROJECT_PATH, "results/attendance_log.csv")
TEST_DATA_DIR = os.path.join(BASE_PROJECT_PATH, "data/test/") 

# Asigurăm existența folderului de rezultate
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

# =========================================================
# 2. LOAD_MODEL (Încărcarea "creierului" antrenat)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceClassifierCNN(num_classes=4)

try:
    # Încărcăm fișierul binar .pt generat la train
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"[SUCCESS] Starea LOAD_MODEL: Model încărcat cu succes din {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Starea LOAD_MODEL a eșuat (Model invalid/Python error): {e}")
    exit()

class_names = {0: "RoundFace", 1: "OvalFace", 2: "LongFace", 3: "Unknown"}

# =========================================================
# 3. PREPROCESS & INFERENCE (Logica de procesare)
# =========================================================
def run_system_test(image_path):
    """Execută fluxul: PREPROCESS -> INFERENCE -> DISPLAY -> LOG."""
    try:
        # --- PREPROCESS ---
        img = cv2.imread(image_path)
        if img is None: 
            raise ValueError("Fișier lipsă sau imagine coruptă")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (64, 64)) # Resize conform diagramei
        
        # Normalizare identică cu cea de la antrenare (Nivel 2)
        tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5 
        
        # --- INFERENCE ---
        with torch.no_grad():
            output = model(tensor.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        # --- DISPLAY_RESULT & LOG_RESULT ---
        res_class = class_names[prediction.item()]
        conf_score = f"{confidence.item():.2%}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Scriere în CSV (Audit Log - Nivel 3)
        file_exists = os.path.isfile(LOG_CSV)
        with open(LOG_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Imagine", "Predictie", "Incredere"])
            writer.writerow([timestamp, os.path.basename(image_path), res_class, conf_score])
            
        print(f"[DISPLAY] {os.path.basename(image_path)} -> {res_class} ({conf_score})")
        return True

    except Exception as e:
        # --- ERROR (Starea de eroare din diagramă) ---
        print(f"[ERROR] Starea ERROR pentru {os.path.basename(image_path)}: {e}")
        return False

# =========================================================
# 4. MONITOR MODE (Simulare sistem industrial)
# =========================================================
def monitor_mode():
    print(f"\n[INFO] Sistem în IDLE. Pune imagini noi în: {TEST_DATA_DIR}")
    processed_files = set(os.listdir(TEST_DATA_DIR)) 

    try:
        while True:
            current_files = set(os.listdir(TEST_DATA_DIR))
            new_files = current_files - processed_files
            
            for file_name in new_files:
                if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    time.sleep(0.5) # Așteptăm scrierea completă pe disc
                    run_system_test(os.path.join(TEST_DATA_DIR, file_name))
                processed_files.add(file_name)
            
            time.sleep(1) # Verificare secundară
    except KeyboardInterrupt:
        print("\n[STOP] Sistem oprit. Eliberare resurse.")

if __name__ == "__main__":
    # La pornire, procesăm lotul inițial din folderul de test
    initial_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.jpg")) + \
                    glob.glob(os.path.join(TEST_DATA_DIR, "*.png"))
    
    print(f"[INFO] Se procesează {len(initial_files)} imagini găsite...")
    for img_path in initial_files:
        run_system_test(img_path)
    
    # Trecem în modul de monitorizare live
    monitor_mode()