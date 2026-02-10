# Biblioteci pentru copiere și amestecare date
import os  # Căi și foldere
import shutil  # Copiere fișiere
import random  # Amestecare reproducibilă cu seed

# =========================================================
# SECȚIUNEA 1: CONFIGURĂRI
# =========================================================
# Detectează rădăcina proiectului
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, "../../"))

# Input: imagini procesate (Om, Cerc, Elipsa, Oval)
INPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
# Output: foldere split (train, validation, test)
OUTPUT_BASE = os.path.join(BASE_DIR, "data")

# Split-uri: 70% antrenament, 15% validare, 15% test
SPLITS = ["train", "validation", "test"]

def split_data():
    """Pipeline: citește clase -> amesteacă imagini -> împarte 70/15/15 -> copie în foldere"""
    # PASUL 1: Detectează clasele (Om, Cerc, Elipsa, Oval)
    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"[INFO] Clase găsite: {classes}")

    # PASUL 2: pentru fiecare clasă
    for class_name in classes:
        class_path = os.path.join(INPUT_DIR, class_name)
        # Filtrare: doar imagini .jpg
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        
        # Amesteacă reproducibil (seed 42 = același rezultat oricând)
        random.seed(42)
        random.shuffle(images)

        # PASUL 3: Calculează proporții (70/15/15)
        total = len(images)
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)
        # Restul la test (total - train - val)

        # Împarte lista în 3 subseturi
        train_list = images[:n_train]
        val_list = images[n_train:n_train + n_val]
        test_list = images[n_train + n_val:]

        print(f"[CLASS {class_name}] Total: {total} | Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")

        # PASUL 4: Copia imagini în folderele corespunzătoare
        for split_name, img_list in zip(SPLITS, [train_list, val_list, test_list]):
            # Creează folder: data/train/Om, data/validation/Om, etc.
            dest_dir = os.path.join(OUTPUT_BASE, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copia fiecare imagine
            for img_name in img_list:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(dest_dir, img_name)
                shutil.copy2(src, dst)  # copy2 = copy + mențin metadata

    print("\n[DONE] Dataset împărțit 70/15/15 în data/train, data/validation și data/test!")

# =========================================================
# SECȚIUNEA 2: EXECUȚIE PRINCIPALĂ
# =========================================================
if __name__ == "__main__":
    # Lansează pipeline de split dataset
    split_data()