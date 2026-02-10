# Modul 2: Arhitectura și Antrenarea Rețelei Neuronale

Acest modul reprezintă gestionează ranziția de la date brute la un model capabil de **clasificare în timp real**.  
Arhitectura este optimizată pentru rulare pe **hardware cu resurse limitate** (laptop / edge computing), menținând o latență minimă.

---

##  Arhitectura Modelelor (`model_torch.py`)

Au fost implementate și comparate **două filozofii de calcul** pentru a valida superioritatea învățării convoluționale asupra procesării liniare a pixelilor.

### CNN_Principal (Arhitectura câștigătoare)

- **3 straturi convoluționale**  
  Utilizate pentru extragerea ierarhică a trăsăturilor:
  - margini
  - forme
  - structuri faciale globale

- **Batch Normalization & Dropout (0.4)**  
  Tehnici de regularizare pentru:
  - prevenirea overfitting-ului
  - stabilitatea gradientului în timpul antrenării

- **Max Pooling**  
  Reducerea dimensiunii spațiale pentru eficiență computațională și robustețe.

---

### MLP_Baseline (Grup de control)

- Rețea densă (**Fully Connected**) utilizată exclusiv ca **baseline comparativ**
- Demonstrează ineficiența procesării liniare a pixelilor față de abordarea spațială specifică CNN-urilor

---

##  Pipeline-ul de Antrenare și Optimizare (`train_model.py`)

Acest script consolidează etapele de **antrenare, evaluare și vizualizare** într-un flux atomic (*end-to-end*), garantând **reproductibilitatea rezultatelor**.

### Funcționalități cheie:

- **Data Augmentation**  
  Integrarea librăriei *Albumentations* pentru simularea:
  - variațiilor de lumină
  - zgomotului de tip ISO
  - distorsiunilor de perspectivă

- **Evaluare automată**  
  Calculul imediat al metricilor:
  - Accuracy
  - F1-Score
  - Confusion Matrix

- **Export ONNX**  
  Conversia modelului din format PyTorch (`.pt`) în format universal **ONNX**, pentru interoperabilitate ridicată.

- **Vizualizare rezultate**  
  Generarea automată a curbelor de învățare în directorul `results/`.

---

##  Validarea în Timp Real (`test_model.py`)

Scriptul de inferență (producție) care simulează un **sistem de monitorizare industrial / campus**.

### Funcționalități:

- **Monitor Mode**  
  Scanează activ folderele de intrare pentru imagini noi.

- **Attendance Logging**  
  Salvează automat rezultatele predicțiilor în `attendance_log.csv`, incluzând:
  - timestamp
  - scorul de confidență al predicției

---

##  Rezumat Performanță (Benchmark)

| Model          | Nr. Parametri | Acuratețe Test | Latență Inference |
|---------------|---------------|----------------|-------------------|
| MLP_Baseline  | 6.29 M        | 83.33%         | ~1.16 ms          |
| CNN_Principal | 2.19 M        | 83.89%         | ~2.71 ms          |

---

##  Concluzie Tehnică

Modelul **CNN_Principal** este de aproximativ **3× mai eficient** din punct de vedere al numărului de parametri, oferind în același timp o **capacitate de generalizare superioară**.  

Această caracteristică este esențială pentru **mediul dinamic al unui campus universitar**, unde robustețea și eficiența sunt critice pentru rularea în timp real pe sisteme cu resurse limitate.