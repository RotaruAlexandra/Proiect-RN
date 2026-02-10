# Importare biblioteci esențiale pentru antrenare deep learning
import sys
import os
import torch  # PyTorch: framework DL
import torch.nn as nn  # Module și straturi neuronale
import torch.optim as optim  # Optimizatori (Adam, SGD, etc.)
from torch.utils.data import DataLoader, Dataset  # Încărcare și batch date
from torchvision import datasets  # Datasets ImageFolder
import numpy as np
import pandas as pd  # Export rezultate în CSV
import json  # Export metrici
import time  # Măsurare latență
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # Metrici evaluare
import matplotlib.pyplot as plt  # Plotare grafice
import seaborn as sns  # Matricea confuziei
import albumentations as A  # Augmentări de imagini

# =========================================================
# SECȚIUNEA 1: CONFIGURĂRI ȘI PATH-URI
# =========================================================
# Setare cale rădăcină și importuri locale
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

# Creere directoare pentru modele, documentație și rezultate
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
for d in [MODEL_DIR, DOCS_DIR, RESULTS_DIR]: os.makedirs(d, exist_ok=True)

# Import clasele de modele din model_torch.py (același director)
from model_torch import FaceClassifierCNN, MLP_Baseline

# =========================================================
# SECȚIUNEA 2: DATASET CU AUGMENTĂRI (ALBUMENTATIONS)
# =========================================================
# Wrapper pentru ImageFolder cu transformări Albumentations
class AlbumentationsDataset(Dataset):
    """Încarcă imagini și aplică augmentări pe-the-fly"""
    def __init__(self, image_folder_dataset, transform=None):
        self.dataset = image_folder_dataset  # Dataset ImageFolder din torchvision
        self.transform = transform  # Augmentări Albumentations

    def __getitem__(self, index):
        # Preluare imagine și label
        image, label = self.dataset[index]
        image = np.array(image)  # Conversie PIL -> NumPy (necesar pentru Albumentations)
        # Aplică augmentări dacă sunt definite
        if self.transform:
            image = self.transform(image=image)["image"]
        # Conversie la tensor PyTorch și normalizare [-1, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image = (image - 0.5) / 0.5  # Normalizare stadard
        return image, label

    def __len__(self):
        return len(self.dataset)

# Augmentări industriale aplicat la antrenament (Nivel 2 - Optimizare)
train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),  # Flip orizontal (50% șanse)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),  # Variație luminos
    A.Perspective(scale=(0.05, 0.15), p=0.5),  # Distorsiune perspectivă
    A.Rotate(limit=25, p=0.5),  # Rotație până la 25 grade
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # Blur Gaussian
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # Zgomot ISO
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Distorsiune grid
    A.GridDropout(ratio=0.3, p=0.3),  # Drop pixeli random
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Color jitter
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.2),  # Umbri random
])

# =========================================================
# SECȚIUNEA 3: ANTRENARE COMPLETĂ - Bucla de învățare cu 10 epoci
# =========================================================
def run_full_training(model_class, model_name, train_dl, val_dl, test_dl, device, class_names):
    """Antrenare model pe 10 epoci cu validare și checkpointing"""
    print(f"\n{'-'*60}")
    print(f"[START] Antrenare Arhitectură: {model_name}")
    print(f"{'-'*60}")
    
    # === INIȚIALIZARE ===
    model = model_class(num_classes=len(class_names)).to(device)  # Creează model și trimite pe device
    criterion = nn.CrossEntropyLoss()  # Loss pentru clasificare multi-clasă
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # Optimizer cu L2 regularizare
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)  # Decay LR

    history = {"train_loss": [], "val_loss": [], "val_acc": []}  # Tracker metrici pe epoci
    best_val_loss = float("inf")  # Pentru salvare best model
    epochs = 10  # Total 10 epoci

    # ===== BUCLA PRINCIPALĂ: 10 EPOCI DE ANTRENARE =====
    for epoch in range(epochs):
        # --- FAZA 1: ANTRENAMENT PE TRAIN SET ---
        model.train()  # Activează dropout și batch norm în training mode
        r_loss = 0.0  # Acumulator pentru loss total
        
        for imgs, lbls in train_dl:  # Parcurge mini-batches (batch_size=8)
            # Trimite imagini și labels pe device (GPU/CPU)
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # *** FORWARD PASS ***
            optimizer.zero_grad()  # Reseteaza gradienti din iteratia anterioară (important!)
            out = model(imgs)  # Predicții: shape (batch_size, num_classes) = (8, 4)
            
            # *** CALCULARE LOSS ***
            loss = criterion(out, lbls)  # CrossEntropy compară logits vs true labels
            
            # *** BACKWARD PASS - BACKPROPAGATION ***
            loss.backward()  # Calculează gradienti ∂L/∂w cu chain rule
            
            # *** UPDATE GREUTĂȚI ***
            optimizer.step()  # Actualizează w_new = w_old - lr * gradient
            
            # Accumula loss pentru média pe epocă
            r_loss += loss.item()
        
        # --- FAZA 2: VALIDARE PE VALIDATION SET ---
        model.eval()  # Dezactivează dropout și batch norm (evaluation mode)
        v_loss, preds, targets = 0.0, [], []  # Inițializări
        
        with torch.no_grad():  # Context manager: dezactivează calculul gradienților (mai rapid + memorie)
            for imgs, lbls in val_dl:  # Parcurge validation mini-batches
                # Trimite pe device
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                # Forward pass DOAR (fără backward - nu calculam gradienti)
                out = model(imgs)  # Predicții pe validation set
                
                # Accumula validation loss
                v_loss += criterion(out, lbls).item()
                
                # Extrage predicții: argmax face clasa cu probabilitate maximă
                preds.extend(out.argmax(1).cpu().numpy())  # Clasa prezisă
                targets.extend(lbls.cpu().numpy())  # True labels
        
        # --- CALCUL METRICI PE ÉPOCĂ ---
        avg_train_loss = r_loss / len(train_dl)  # Loss mediu pe antrenament
        avg_v_loss = v_loss / len(val_dl)  # Loss mediu pe validare
        val_acc = accuracy_score(targets, preds)  # % predicții corecte pe validare
        
        # Salvează în istoric pentru grafic
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_v_loss)
        history["val_acc"].append(val_acc)
        
        # --- AFIȘARE LOG PE CONSOLĂ ---
        print(f"Epoch {epoch+1}/10 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_v_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- CHECKPOINTING: SALVEAZĂ CEL MAI BUN MODEL ---
        # Dacă validation loss scade, salvează model (early stopping indirect)
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss  # Update best loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_{model_name}.pt"))  # Salvează greutăți
            print("   ✓ Model salvat!")  # Notificare
            
        # --- LEARNING RATE SCHEDULING ---
        # Dacă validation loss stagneaza (nu scade 3 epoci consecutive),
        # reduce learning rate cu factor=0.3 (lr_new = 0.3 * lr_old)
        scheduler.step(avg_v_loss)

    # ===== EVALUARE FINALĂ: TEST SET ===== (Nivel 2)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"best_{model_name}.pt")))  # Încarcă cel mai bun model
    model.eval()  # Dezactivează dropout și batch norm
    
    # --- MĂSURARE LATENȚĂ INFERENȚĂ ---
    dummy_input = torch.randn(1, 3, 64, 64).to(device)  # Dummy input pentru benchmark
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)  # Warm-up (cache preallocate)
        start_bench = time.time()  # Start cronometru
        for _ in range(100): _ = model(dummy_input)  # 100 forward passes
        latency = ((time.time() - start_bench) / 100) * 1000  # Rezultat în milisecunde

    # --- PREDICȚII PE TEST SET ---
    test_preds, test_true = [], []
    with torch.no_grad():  # Fără calculul gradienților
        for imgs, lbls in test_dl:  # Parcurge test batches
            out = model(imgs.to(device))  # Forward pass
            test_preds.extend(out.argmax(1).cpu().numpy())  # Clasa cu prob max
            test_true.extend(lbls.numpy())  # True labels

    # Calculare metrici - Acuratețe și F1 Score
    acc = accuracy_score(test_true, test_preds)  # % predicții corecte
    f1 = f1_score(test_true, test_preds, average='macro')  # F1 balansat

    # Salvare grafice Loss (Nivel 2)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title(f"Loss Curve - {model_name}")
    plt.savefig(os.path.join(DOCS_DIR, f"loss_curve_{model_name}.png"))
    plt.close()

    # Salvare Matrice Confuzie & ONNX doar pentru modelul principal (Nivel 3)
    if model_name == "CNN_Principal":
        torch.onnx.export(model, dummy_input, os.path.join(MODEL_DIR, "final_model.onnx"))
        cm = confusion_matrix(test_true, test_preds)
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.savefig(os.path.join(DOCS_DIR, f"confusion_matrix_{model_name}.png")); plt.close()

    # Creare dictionar metrici pentru JSON
    res_dict = {
        "Arhitectura": model_name,
        "Acuratete_Test": float(acc),
        "F1_Score_Macro": float(f1),
        "Latenta_ms": float(latency),
        "Numar_Parametri": sum(p.numel() for p in model.parameters())
    }
    
    return res_dict

# =========================================================
# SECȚIUNEA 4: EXECUȚIE PRINCIPALĂ - Orchestrare antrenare
# =========================================================
if __name__ == "__main__":
    # Setare device (GPU dacă disponibil, altfel CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(ROOT_DIR, "data")
    
    # Încărcare seturi de date din JPG (Nivel 1)
    raw_train = datasets.ImageFolder(root=os.path.join(data_path, "train"))
    raw_val = datasets.ImageFolder(root=os.path.join(data_path, "validation"))
    raw_test = datasets.ImageFolder(root=os.path.join(data_path, "test"))
    
    # Creare DataLoaders cu batching
    train_dl = DataLoader(AlbumentationsDataset(raw_train, train_augmentations), batch_size=8, shuffle=True)  # Augmentări pe antrenament
    val_dl = DataLoader(AlbumentationsDataset(raw_val), batch_size=8)  # Fără augmentări
    test_dl = DataLoader(AlbumentationsDataset(raw_test), batch_size=8)  # Fără augmentări
    
    results_summary = []  # Colectare rezultate ambelor modele
    
    # Antrenare ambele arhitecturi (Nivel 3 Bonus: Comparare)
    results_summary.append(run_full_training(FaceClassifierCNN, "CNN_Principal", train_dl, val_dl, test_dl, device, raw_train.classes))
    results_summary.append(run_full_training(MLP_Baseline, "MLP_Baseline", train_dl, val_dl, test_dl, device, raw_train.classes))

    # 1. Export CSV pentru vizualizare rapidă
    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(RESULTS_DIR, "comparare_arhitecturi.csv"), index=False)
    
    # 2. Export JSON pentru audit și integrare
    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n[SUCCESS] Toate fișierele au fost generate în results/ și docs/!")
    print(df)  # Afișare tabel final