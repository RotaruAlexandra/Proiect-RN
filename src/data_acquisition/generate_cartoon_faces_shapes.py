# Importare biblioteci pentru procesare imagini
import cv2  # OpenCV: desenare formelor geometrice și salvare imagini
import numpy as np  # Manipulare matrici pentru crearea imaginilor
import os  # Creare directoare și gestionare căi
import random  # Generare valori random pentru variație în imagini

# ============================================================
# SETĂRI CALE ȘI PARAMETRI
# ============================================================
OUTPUT_DIR = os.path.join("data", "generated")  # Folderul unde se vor salva imaginile generate
IMG_SIZE_RAW = 400  # Dimensiune imagine: 400x400 pixeli
NUM_IMAGES = 300  # Numărul de imagini pe clasă (300 * 3 = 900 total)

# Funcție helper: generează culoare RGB aleatoare pentru variație în imagini
def rand_color():
    # Culori în interval [20-100] pentru gama de griuri/castanii
    return (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))

# Funcție: desenează trăsături faciale pe imagine (păr, urechi, ochi, gură)
def draw_face_features(img, cx, cy, rx, ry):
    # Center (cx, cy) și radii (rx, ry) sunt pozițiile și dimensiunile bazate pe forma principală
    
    # PĂRUL: elipsă deasupra capului
    hair_color = rand_color()
    cv2.ellipse(img, (cx, cy - int(ry * 0.8)), 
                (int(rx * 1.1), int(ry * 0.5)), 0, 0, 360, hair_color, -1)

    # URECHILE: cercuri albe cu contur negru
    cv2.circle(img, (cx - rx - 5, cy), 20, (255, 224, 189), -1)  # Ureche stâng
    cv2.circle(img, (cx + rx + 5, cy), 20, (255, 224, 189), -1)  # Ureche dreapt
    cv2.circle(img, (cx - rx - 5, cy), 20, (0, 0, 0), 1)  # Contur
    cv2.circle(img, (cx + rx + 5, cy), 20, (0, 0, 0), 1)  # Contur

    # OCHII: albi cu pupile negre
    eye_y = cy - int(ry * 0.15)  # Poziție pe verticală
    cv2.ellipse(img, (cx - 35, eye_y), (12, 18), 0, 0, 360, (255, 255, 255), -1)  # Alb stâng
    cv2.ellipse(img, (cx + 35, eye_y), (12, 18), 0, 0, 360, (255, 255, 255), -1)  # Alb dreapt
    cv2.circle(img, (cx - 35, eye_y), 6, (0, 0, 0), -1)  # Pupilă stâng
    cv2.circle(img, (cx + 35, eye_y), 6, (0, 0, 0), -1)  # Pupilă dreapt

    # GURA: semicerc negru
    mouth_y = cy + int(ry * 0.4)  # Poziție jos pe față
    cv2.ellipse(img, (cx, mouth_y), (35, 15), 0, 0, 180, (0, 0, 0), 2)
    
    return img

# Funcție: generează o imagine cu o formă geometrică și trăsături faciale
# Input: class_name = "Cerc", "Elipsa" sau "Oval"
def generate_shape(class_name):
    # Inițializează imagine: fond alb (RGB: 255, 255, 255)
    img = np.full((IMG_SIZE_RAW, IMG_SIZE_RAW, 3), 255, dtype=np.uint8)
    cx, cy = IMG_SIZE_RAW // 2, IMG_SIZE_RAW // 2  # Centrul imaginii
    skin_color = (255, 224, 189)  # Culoare piele: bej
    
    if class_name == "Cerc":
        # Cerc perfect: rază egală pe ambele axe
        r = random.randint(100, 120)  # Variație în dimensiune
        cv2.circle(img, (cx, cy), r, skin_color, -1)  # Cerc plin
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)  # Contur negru
        img = draw_face_features(img, cx, cy, r, r)  # Trăsături faciale simetrice
        
    elif class_name == "Elipsa":
        # Elipsă verticală: vAxis > hAxis (mai înalt decât lat)
        rx, ry = random.randint(90, 105), random.randint(130, 145)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, skin_color, -1)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 0), 2)
        img = draw_face_features(img, cx, cy, rx, ry)
        
    elif class_name == "Oval":
        # Oval asimetric: jumătate superioară mai mică, jumătate inferioară mai mare
        rx = random.randint(90, 105)
        ry_top, ry_bot = random.randint(110, 125), random.randint(140, 155)
        # Jumătate superioară (mai îngustă)
        cv2.ellipse(img, (cx, cy), (rx, ry_top), 0, 180, 360, skin_color, -1)
        # Jumătate inferioară (mai lată)
        cv2.ellipse(img, (cx, cy), (rx, ry_bot), 0, 0, 180, skin_color, -1)
        # Contururi
        cv2.ellipse(img, (cx, cy), (rx, ry_top), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(img, (cx, cy), (rx, ry_bot), 0, 0, 180, (0, 0, 0), 2)
        img = draw_face_features(img, cx, cy, rx, (ry_top + ry_bot)//2)  # Trăsături cu radius mediu

    return img

# ====== EXECUȚIA PRINCIPALĂ: Generează și salvează imaginile ======
classes = ["Cerc", "Elipsa", "Oval"]  # Cele 3 clase
for cls in classes:
    path = os.path.join(OUTPUT_DIR, cls)  # Cale pe clasă
    if not os.path.exists(path):
        os.makedirs(path)  # Creează folder dacă nu există
    
    print(f"Generez poze pentru {cls}...")
    # Generează NUM_IMAGES imagini pe clasă
    for i in range(1, NUM_IMAGES + 1):
        image = generate_shape(cls)  # Creează imagine
        cv2.imwrite(os.path.join(path, f"{cls}_{i}.jpg"), image)  # Salvează cu nume unic

# Notificare finalizare
print(f"\n[GATA] Cele {len(classes) * NUM_IMAGES} de poze au fost salvate în: {OUTPUT_DIR}")