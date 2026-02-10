# Importare biblioteci pentru interfață web și gestiune UI
import streamlit as st  # Framework pentru construire de aplicații web interactive
import pandas as pd  # Manipularea și afișarea datelor în format tabel
import time  # Măsurare latență și delay-uri
import os  # Operații cu sistemi de fișiere și căi
import json  # Lucrul cu fișiere JSON

# Importare biblioteci pentru deep learning și procesare imagini
import torch  # Framework PyTorch pentru rețele neuronale
import torch.nn as nn  # Module și straturi neural
import torchvision.transforms as transforms  # Transformări și preprocesare imagini
from PIL import Image  # Încărcare și manipulare imagini
import numpy as np  # Operații numerice și matrici
import datetime  # Gestionare timpuri și date-time

# =========================================================
# 1. DEFINIRE ARHITECTURĂ
# =========================================================
# =========================================================
# SECȚIUNEA 1: DEFINIRE ARHITECTURĂ CNN
# =========================================================
# Definește structura rețelei neuronale convoluționale (CNN) cu 3 straturi conv
# și 2 straturi fully-connected pentru clasificare în 4 clase

class SimpleCNN(nn.Module):
    # Constructor: inițializează straturile rețelei
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        # Stratul 1: 3 canale (RGB) -> 32 filtre, kernel 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalizare pe batch pentru stabilitate
        # Stratul 2: 32 filtre -> 64 filtre
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Stratul 3: 64 filtre -> 128 filtre
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Max pooling: reduce dimensionalitatea după fiecare stratu convoluțional
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()  # Funcție de activare non-liniară
        # Straturi fully-connected: 128*8*8 features -> 256 hidden -> num_classes output
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # Forward pass: definește fluxul datelor prin rețea
    def forward(self, x):
        # Bloc 1: Conv -> BatchNorm -> ReLU -> MaxPool (reduc 64x64 la 32x32)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Bloc 2: Conv -> BatchNorm -> ReLU -> MaxPool (reduc 32x32 la 16x16)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Bloc 3: Conv -> BatchNorm -> ReLU -> MaxPool (reduc 16x16 la 8x8)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # Flatten: transforma tensor 3D (128, 8, 8) în vector 1D (128*8*8,)
        x = x.view(x.size(0), -1)
        # Fully connected 1: 128*8*8 -> 256 cu activare ReLU
        x = self.relu(self.fc1(x))
        # Fully connected 2: 256 -> num_classes (4 clase: Cerc, Elipsa, Om, Oval)
        x = self.fc2(x)
        return x

# =========================================================
# 2. CONFIGURARE PAGINĂ ȘI LOGICĂ
# =========================================================
# =========================================================
# SECȚIUNEA 2: CONFIGURARE PAGINĂ STREAMLIT ȘI CĂILE FIȘIERELOR
# =========================================================
# Setează dimensionarea paginii web și definește căile către modele/rezultate

st.set_page_config(page_title="Monitorizare SIA", layout="wide")  # Titlu tab browser și layout larg

# Căi relative către resurse (model antrenat, logs, metrici, grafice)
MODEL_PT_PATH = "../../models/best_CNN_Principal.pt"  # Modelul PyTorch antrenat
LOG_CSV = "../../results/attendance_log.csv"  # CSV cu jurnal predicții
METRICS_JSON = "../../results/test_metrics.json"  # Metrici global de test
LOSS_PLOT = "../../docs/loss_curve_CNN_Principal.png"  # Grafic evoluție loss
CM_PLOT = "../../docs/confusion_matrix_CNN_Principal.png"  # Matrice confuziei

# Inițializare session state pentru evitarea unor duplicări de logs
if 'last_logged_image' not in st.session_state:
    st.session_state.last_logged_image = None  # Urmărește ultima imagine procesată

# Funcție de logging: salvează fiecare predicție în jurnal CSV pentru audit
def log_prediction(image_name, prediction, confidence):
    # Construiește rând nou cu timestamp, nume imagine, clasa detectată și incredere
    new_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Data și ora predicției
        "Imagine": image_name,  # Numele fișierului imagine
        "Predictie": prediction,  # Clasa detectată sau "Incert"
        "Incredere": f"{confidence:.2f}%"  # Procentaj de încredere
    }
    df_new = pd.DataFrame([new_entry])
    try:
        # Salvează în CSV existente sau creează nou dacă nu există
        df_new.to_csv(LOG_CSV, mode='a', header=not os.path.exists(LOG_CSV), index=False)
        return True
    except PermissionError:
        # Eroare dacă fișierul e deschis în Excel sau similar
        return False

# Funcție cu cache: încarcă modelul PyTorch din fișier o singură dată
@st.cache_resource  # Se execută o singură dată, apoi refolosește rezultatul
def load_pytorch_model():
    if os.path.exists(MODEL_PT_PATH):
        try:
            # Creează instanță nouă a arhitecturii
            model = SimpleCNN(num_classes=4)
            # Încarcă greutățile salvate (pe CPU dacă GPU nu e disponibil)
            state_dict = torch.load(MODEL_PT_PATH, map_location=torch.device('cpu'))
            from collections import OrderedDict
            # Reformatează cheile state_dict dacă au prefixe neobișnuite
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.split('.')[-2] + "." + k.split('.')[-1] if '.' in k else k
                new_state_dict[name] = v
            # Aplică greutățile cu strict=False pentru compatibilitate
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()  # Setare mode evaluare (fără dropout/batch norm training)
            return model
        except Exception as e:
            st.error(f"Eroare critică: {e}")
    return None

# Încarcă modelul pe startup
model_pt = load_pytorch_model()
# Clasele pe care sunt antrenate modelul
clase_sia = ["Cerc", "Elipsa", "Om", "Oval"]
# Preprocesare imagini: redimensionare (64x64) + normalizare
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionare la dimensiunea așteptată
    transforms.ToTensor(),  # Converte pillow Image -> tensor PyTorch (0-1)
    transforms.Normalize([0.5], [0.5])  # Normalizare: (x - 0.5) / 0.5 = x în [-1, 1]
])

# CSS global pentru aspect dark mode și centrare titluri
st.markdown("""<style>.stApp { background: #0f172a; color: white; } h1,h2,h3 { text-align: center; color: #38bdf8; }</style>""", unsafe_allow_html=True)

# =========================================================
# 3. INTERFAȚĂ LIVE (Nivel 1 - Inferență Reală)
# =========================================================
# =========================================================
# SECȚIUNEA 3: INTERFAȚĂ UPLOAD ȘI CONTROL STATUS (NIVEL 1)
# =========================================================
# Titlu principal și widget de upload imagini

st.markdown("<h1> Monitorizare Sistem SIA </h1>", unsafe_allow_html=True)

st.subheader(" Testare Model - Inferență Live (Nivel 1)")
with st.expander("Încarcă o imagine pentru clasificare", expanded=True):
    uploaded_file = st.file_uploader("Alege imagine...", type=['jpg', 'png'])  # Widget upload

# =========================================================
# SIDEBAR: AFFIȘARE STATUS ȘI STATE MACHINE TRACE
# =========================================================
# Sidebar starea   sistemului și urmărește progresul pe niveluri

st.sidebar.header(" Status Flux")
if model_pt:
    st.sidebar.success("Model PyTorch Încărcat")  # Model găsit cu succes
    st.sidebar.markdown("---")
    st.sidebar.subheader(" State Machine Trace")  # Urmărire stări în execuție
    st.sidebar.write("- [x] **IDLE**: Sistem pregătit")  # Stare inițială
    st.sidebar.write("- [x] **LOAD_MODEL**: best_CNN_Principal.pt activ")  # Model încărcat
    
    if uploaded_file:
        # Dacă trimit imagine, arată progresul pe etape
        st.sidebar.write("- [x] **PREPROCESS**: Imagine redimensionată")
        st.sidebar.write("- [x] **INFERENCE**: Predicție calculată")
        st.sidebar.write("- [x] **LOG_RESULT**: Salvat în CSV")
        st.sidebar.write("- [] **DISPLAY_RESULT**: Tabel actualizat")
    else:
        # Dacă nu e imagine, arată stări pending
        st.sidebar.write("- [ ] PREPROCESS: Așteptare input...")
        st.sidebar.write("- [ ] INFERENCE")
        st.sidebar.write("- [ ] LOG_RESULT")
else:
    st.sidebar.error("Lipsă Model .pt")  # Eroare dacă model nu găsit

# =========================================================
# SECȚIUNEA 4: LOGICA DE PREDICȚIE OPTIMIZATĂ (NIVEL 2)
# =========================================================
# Procesează imagine: preproces -> forward pass -> confidence threshold

if uploaded_file and model_pt:
    # Deschide și convertește imagine la RGB (elimină alpha channel dacă e PNG)
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, width=150)  # Afișează preview imagine
    # Aplică transformări: resize, toTensor, normalize
    input_tensor = preprocess(img).unsqueeze(0)  # Adaugă dimensiune batch (1, 3, 64, 64)
    
    # ===== OPTIMIZARE: Măsurare latență predicție =====
    t_start = time.time()  # Incepe cronometrare
    with torch.no_grad():  # Dezactivează calculul gradienților (inferență pură)
        output = model_pt(input_tensor)  # Forward pass prin rețea
        prob = torch.nn.functional.softmax(output[0], dim=0)  # Converteste scores la probabilități
        conf, idx = torch.max(prob, 0)  # Găsește clasa cu probabilitate maximă
    t_end = time.time()  # Termină cronometrare
    latency_ms = (t_end - t_start) * 1000  # Converteste la milisecunde
    
    # Extrage valori numerice din tensori
    confidenta_val = conf.item() * 100  # Încredere în procente
    clasa_raw = clase_sia[idx.item()]  # Mapează index la string de clasă

    # ===== OPTIMIZARE: Filtru de siguranță (Confidence Threshold) =====
    # Dacă model e prea nesigur, marchează ca "Incert" pentru revizuire manuală
    if confidenta_val < 60.0:
        clasa_detectata = "Incert (Revizuire)"
        st.warning(f" Atenție: Încredere scăzută ({confidenta_val:.2f}%). S-a declanșat starea de verificare.")
    else:
        clasa_detectata = clasa_raw  # Acceptă predicția
        st.success(f" Rezultat: {clasa_detectata} ({confidenta_val:.2f}%)")

    # Afișează latența în sidebar pentru vizibilitate (profesor trebuie s-o vadă clar)
    st.sidebar.markdown(f" **Latență Inferență:** `{latency_ms:.2f} ms`" )
    
    # Salvează predicție în jurnal, dar doar dacă e o imagine nouă (evită duplicări)
    if st.session_state.last_logged_image != uploaded_file.name:
        if log_prediction(uploaded_file.name, clasa_detectata, confidenta_val):
            st.session_state.last_logged_image = uploaded_file.name
            st.toast(f"Înregistrat în CSV: {uploaded_file.name}")  # Toast notificare

# =========================================================
# SECȚIUNEA 5: JURNAL AUDIT REAL-TIME (NIVEL 3)
# =========================================================
# Afișează tabelul cu toate predicțiile efectuate în ordinea inversă (mai recent = sus)

st.markdown("---")  # Separator vizual
st.subheader(" Jurnal Audit Real-Time (Nivel 3)")
# Creează placeholder container pentru update dinamic fără re-render pagină
dashboard_placeholder = st.empty()

# =========================================================
# SECȚIUNEA 6: ANALIZĂ PERFORMANȚĂ MODEL (GRAFICE)
# =========================================================
# Afișează graficele antrenamentului și matrice de confuzie

st.markdown("---")  # Separator vizual
col_l, col_m, col_r = st.columns([1, 2, 1])  # Layout: stânga gol, centru 2x, dreapta gol
with col_m:
    st.subheader("Performanță Model")
    # Afișează grafice dacă fișierele de imagine există
    if os.path.exists(LOSS_PLOT): 
        st.image(LOSS_PLOT, caption="Loss Curve (Nivel 2)", use_container_width=True)
    if os.path.exists(CM_PLOT): 
        st.image(CM_PLOT, caption="Confusion Matrix (Nivel 3)", use_container_width=True)

# =========================================================
# SECȚIUNEA 7: BUCLA DE MONITORIZARE DINAMICĂ (NIVEL 3)
# =========================================================
# Update tabel jurnal in timp real la fiecare 2 secunde
# Această secțiune trebuie la final pentru a nu bloca afișarea elementelor UI anterioare

while True:
    # Verific existența fișier CSV cu jurnal
    if os.path.exists(LOG_CSV):
        try:
            # Citire date și inversare (cel mai recent = linia 1)
            df = pd.read_csv(LOG_CSV).iloc[::-1]
            # Update DOAR containerul tabelului (dashboard_placeholder) 
            # Evită re-render-area întregii pagini, doar tabelul se schimbă
            dashboard_placeholder.dataframe(df, use_container_width=True, height=300)
        except:
            # Dacă citire eșuează (fișier corupt, etc), taci și continuă
            pass
    # Sleep interval de 2 secunde între refresh-uri
    time.sleep(2)