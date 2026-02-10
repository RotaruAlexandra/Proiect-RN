# Biblioteci pentru procesare imagini
import os  # Căi și foldere
import cv2  # Citire, resize și salvare imagini
import numpy as np  # Operații pe matrici

# ============================================================
# SECȚIUNEA 1: CONFIGURĂRI
# ============================================================
# Detect rădăcina proiectului automat
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Input: imagini brute (Cerc, Elipsa, Oval)
INPUT_DIR = os.path.join(BASE_DIR, "data", "validation")

# Output: imagini resize 64x64 (compatibil CNN)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

TARGET_SIZE = (64, 64)  # Rezoluție finală

def preprocess_image(img_path):
    """Citire și resize la 64x64"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Nu pot citi: {img_path}")
        return None

    # Resize: 64x64 pt RN
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def run_preprocessing():
    """Pipeline: iterează clase (Cerc, Elipsa, Oval) -> procesează imagini -> salvează"""
    print(f"[INFO] Citesc din: {INPUT_DIR}")
    
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Folderul de intrare nu există!")
        return

    # Pentru fiecare clasă (Cerc, Elipsa, Oval)
    for class_name in os.listdir(INPUT_DIR):
        class_folder = os.path.join(INPUT_DIR, class_name)

        if not os.path.isdir(class_folder):
            continue

        print(f"\n[CLASS] Procesez clasa: {class_name}")

        # Creează folder output pentru clasă
        dest_class_folder = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_class_folder, exist_ok=True)

        # Filtrare: doar imagini (jpg, png)
        images = [f for f in os.listdir(class_folder) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Procesează fiecare imagine
        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            
            processed_img = preprocess_image(img_path)  # Resize
            
            if processed_img is not None:
                # Salvează cu extensie .jpg
                out_path = os.path.join(dest_class_folder, img_name)
                
                if not out_path.lower().endswith(".jpg"):
                    out_path = os.path.splitext(out_path)[0] + ".jpg"
                
                cv2.imwrite(out_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[OK] Salvat: {img_name} -> {class_name}")

    print(f"\n[FINISHED] Toate figurile au fost salvate în: {OUTPUT_DIR}")

# ============================================================
# SECȚIUNEA 2: EXECUȚIE
# ============================================================
if __name__ == "__main__":
    # Lansează pipeline complet de preprocesare
    run_preprocessing()