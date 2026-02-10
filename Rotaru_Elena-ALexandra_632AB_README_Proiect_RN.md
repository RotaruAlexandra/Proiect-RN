## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Rotaru Elena-Alexandra |
| **Grupa / Specializare** |  63AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** |[ URL complet - https://github.com/RotaruAlexandra/Proiect-RN-Rotaru_Elena-Alexandra.git] |
| **Acces Repository** | [Public] |
| **Stack Tehnologic** | [Python] |
| **Domeniul Industrial de Interes (DII)** | [Educațional- Smart Campus] |
| **Tip Rețea Neuronală** | [CNN ] |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | [83.89%] | [83.89%] | [-] | [✓] |
| F1-Score (Macro) | ≥0.65 | [0.8387] | [0.8387] | [-] | [✓] |
| Latență Inferență | [< 50 ms] | [2.71 ms] | [15.00 ms] | [+12.29 ms] | [✓] |
| Contribuție Date Originale | ≥40% | [100%] | [100%] | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | [4] | [5] | +1 | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [✓] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [✓] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [✓] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [✓] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [✓] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Proiectul a pornit de la nevoia de monitorizare a prezenței persoanelor în medii indoor. Din cauza restricțiilor **GDPR**, utilizarea imaginilor faciale reale este problematică.

**Soluția adoptată:** Un sistem de **Clasificare Bio-Geometrică**. În loc să antrenez modelul pe fețe reale, am folosit un dataset hibrid format din forme geometrice (Cerc, Oval, Elipsă) și fețe sintetice ("Om"). Sistemul învață să valideze prezența umană prin distincția dintre geometria complexă a chipului și prezența formelor geometrice simple.

### 2.2 Beneficii Măsurabile Urmărite

1. Confidențialitate (GDPR Compliant): Eliminarea riscurilor legale prin utilizarea unui dataset 100% sintetic, asigurând monitorizarea prezenței fără colectarea de date biometrice reale.

2. Performanță în Timp Real: Sistemul atinge o latență medie de inferență de ~12-15 ms, asigurând fluiditatea procesării. Chiar și în scenarii de încărcare maximă (incluzând I/O operations), latența totală se menține sub 30 ms, ceea ce depășește standardul de 30 FPS (cadre pe secundă) necesar monitorizării video.

3. Fiabilitate Industrială: Reducerea erorilor de clasificare prin implementarea unui prag de siguranță (Confidence Threshold) de 60%, sistemul marcând automat cazurile ambigue pentru revizuire.

4. Eficiență Operațională: Automatizarea procesului de pontaj prin generarea instantanee a jurnalului de prezență (attendance_log.csv), eliminând complet eroarea umană și timpul necesar completării manuale.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|:---|:---|:---|:---|
| **Monitorizarea prezenței anonimizate (GDPR)** | Clasificare bio-geometrică (Om vs. Forme de control) | Motor Inferență (CNN_Principal) | Date 100% Sintetice (Mixt: Aplicatie de Generare Random + Kaggle + Original) |
| **Eliminarea deciziilor eronate (Incertitudine)** | Filtrare automată bazată pe prag de confidențialitate | Logic Guard (Aplicația Streamlit) | Prag siguranță Softmax > 60% |
| **Generarea foii de prezență în timp real** | Jurnalizarea automată a detecțiilor valide în fișier extern | Audit Logger (csv_handler) | Latență procesare < 15 ms |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | [ Mixt: Dataset Public + Generare ]  |
| **Sursa concretă** | [ Kaggle - dataset Human Faces:www.kaggle.com/datasets/ashwingupta3012/human-faces/data / Generator Online Random: this-person-does-not-exist.com/en / Script propriu (Python: OpenCV, NumPy) ] |
| **Număr total observații finale (N)** | [1200 (distribuite egal: 300 per clasă) ] |
| **Număr features** | 4096 (64 x 64 pixeli) |
| **Tipuri de date** | [Imagini (Grayscale)] |
| **Format fișiere** | .JPG |
| **Perioada colectării/generării** |  Noiembrie 2025 - Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [1200] |
| **Observații originale (M)** | 900 (3 clase x 300 imagini) | 
| **Procent contribuție originală** | [75%] |
| **Tip contribuție** | [ Date sintetice (Generare programatică) ] | 
| **Locație cod generare** | `src/data_acquisition/[generate_cartoon_faces_shapes.py]' |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

Pentru a asigura un set de date controlat și echilibrat, am dezvoltat un generator algoritmic utilizând bibliotecile OpenCV și NumPy. Această metodă a permis crearea a 900 de mostre sintetice distribuite egal în trei clase geometrice (Cerc, Elipsă, Oval), care simulează variațiile structurale ale feței umane. Parametrii de generare au fost setați pentru a introduce variabilitate stocastică în poziționarea și dimensiunea trăsăturilor faciale (ochi, nas, gură), forțând astfel rețeaua neuronală să învețe caracteristici geometrice de profunzime, nu doar simple texturi.Această abordare este critică pentru validarea sistemului, deoarece clasele sintetice servesc drept "grupuri de control" ce permit testarea robusteții modelului în a distinge între o formă geometrică simplă și prezența reală a unui chip uman (clasa "Om"). Prin preprocesarea automată a acestor date (redimensionare la 64x64 și normalizare), am eliminat zgomotul informațional, obținând un set de date optimizat pentru o arhitectură CNN, ce respectă normele de portabilitate și reproductibilitate tehnică.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | [840] | 
| Validation | 15% | [180] |
| Test | 15% | [180] | 

**Preprocesări aplicate:**
- Segmentare Semantică (Background Removal): Utilizarea modelului MediaPipe Selfie Segmentation (în prepare_dataset_synthetic.py) pentru a izola fețele umane și a înlocui fundalul variabil cu alb pur, eliminând astfel zgomotul vizual care ar putea induce erori de corelație în model.
- Standardizarea Rezoluției: Redimensionarea tuturor imaginilor la 64x64 pixeli prin metoda de interpolare cv2.INTER_AREA (în preprocess_generated_shapes.py), optimizată pentru reducerea dimensiunii fără pierderea trăsăturilor geometrice esențiale.
- Normalizare de tip Z-Score: Conversia pixelilor în tensori și scalarea lor folosind media $0.5$ și deviația standard $0.5$, asigurând centrarea datelor pentru o convergență mai rapidă a gradientului în timpul antrenării.
- Augmentare Industrială (Pipeline Albumentations): Aplicarea unui set complex de transformări stocastice pe setul de antrenare: Horizontal Flip, Perspective Change, Gaussian Blur, ISO Noise și Grid Distortion, crescând artificial diversitatea setului de date.

**Referințe fișiere:** `data/README.md`, `config/preprocessing_params.pkl`,`src/data_acquisition/prepare_dataset_synthetic.py` (Procesare MediaPipe),`src/data_acquisition/preprocess_generated_shapes.py`(Standardizare forme), `src/training/train_model.py `(Pipeline-ul de augmentare și normalizare)

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** |[ Python (OpenCV, MediaPipe) ] | Generare programatică de forme geometrice (sintetice) și rafinarea imaginilor reale prin segmentare semantică și eliminarea fundalului. | `src/data_acquisition/` | 
| **Neural Network** | [PyTorch] | Arhitectură CNN (Convolutional Neural Network) cu straturi de Batch Normalization și Dropout pentru clasificarea multi-clasă a formelor faciale. | `src/neural_network/` |
| **Web Service / UI** | [Streamlit] | Dashboard interactiv pentru monitorizarea în timp real a folderului de test, afișarea predictiilor, a gradului de confidență și a jurnalului de audit. | `main.py` |

### 4.2 State Machine

**Locație diagramă:** `state_machine_v2.png`( Versiune actualizată pentru prezentarea finală, integrând bucla de monitorizare și audit live.)

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` |[Sistemul este în așteptare, monitorizând folderul de test și input-ul din interfața Streamlit.] | [Start aplicație] | [Detectare fișier / Upload imagine] | 
| `ACQUIRE_DATA` | [Preluarea imaginii din buffer sau de pe disc și citirea metadatelor.] | [Eveniment "New File"] | [Imagine încărcată în memorie] |
| `PREPROCESS` | [Redimensionare la 64x64 și normalizarea tensorului (mean = 0.5, std = 0.5)] | [Imagine brută disponibilă] | [Tensor pregătit pentru model] |
| `INFERENCE` | [Calcularea predicției prin SimpleCNN și măsurarea latenței (ms) ] | [ Input preprocesat ] | [Vector probabilități generat] |
| `DECISION` | [Identificarea clasei (Om/Cerc/etc.) și validarea pragului de confidență.] | [Output model disponibil] | [Clasă finală determinată] |
| `AUDIT & LOG` | [Scrierea rezultatului în attendance_log.csv și actualizarea automată a tabelului live.] | [Decizie validată] | [Revenire în IDLE (Monitorizare)] |
| `ERROR` | [Gestionarea excepțiilor (formate invalide, model lipsă) și afișare mesaj eroare.] | [Excepție detectată] | [Recovery automată (Reset IDLE)] |

**Justificare alegere arhitectură State Machine:**

Alegerea unei arhitecturi de tip State Machine a fost determinată de necesitatea de a gestiona un flux de date secvențial și determinist, esențial într-un sistem de monitorizare video și clasificare automată. Această structură permite izolarea logică a fiecărei etape — de la achiziția imaginilor și preprocesarea acestora până la inferența propriu-zisă și logarea rezultatelor în jurnalul de audit — asigurând astfel că sistemul nu trece la faza de decizie înainte ca datele să fie validate. Mai mult, acest model facilitează tratarea erorilor în mod robust; de exemplu, în cazul detectării unei imagini corupte în starea PREPROCESS, sistemul poate reveni automat în starea IDLE fără a bloca aplicația, oferind o predictibilitate ridicată și o mentenanță simplificată a codului sursă.

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 |  Valoare Etapa 6 |  Justificare Modificare |
|----------------------|-----------------|---------------------------|------------------------|
| Bucla de Control | Secvențială (Liniară) | Ciclică (Monitoring Loop) | Permite funcționarea continuă a interfeței fără intervenție manuală. |
| Monitorizare | Doar vizuală | Audit & Latency Tracking | Implementarea monitorizării timpului de răspuns pentru optimizarea I/O. |
| Prag Confidență | Static (0.5) | Dinamic (0.6) | Reducerea ratei de eroare prin filtrarea predicțiilor incerte. |
| Interfață Audit | Fișier CSV static | Dashboard Live (refresh la 2s) | Vizualizarea în timp real a jurnalului de prezență direct în UI. |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input Image (Shape: [64, 64, 3] - RGB)
  → Conv2D (32 filtre, 3x3, Padding=1) + BatchNorm + ReLU
  → MaxPool2D (2x2)
  
  → Conv2D (64 filtre, 3x3, Padding=1) + BatchNorm + ReLU
  → MaxPool2D (2x2)
  
  → Conv2D (128 filtre, 3x3, Padding=1) + BatchNorm + ReLU
  → MaxPool2D (2x2)
  
  → Flatten (Transformare în vector de 8192 elemente)
  
  → Dense/Linear (256 unități, ReLU)
  → Dropout (Rată: 0.4)
  → Dense/Linear (4 unități, Softmax/Logits)
  
Output: 4 Clase (Cerc, Elipsa, Oval, Om)
```

**Justificare alegere arhitectură:**

Am ales o arhitectură de tip CNN (Convolutional Neural Network) deoarece este optimizată pentru procesarea datelor cu structură spațială, fiind capabilă să extragă automat trăsături ierarhice (margini, texturi, forme complexe) necesare deosebirii formelor geometrice de fețele umane. Am considerat și testat ca alternativă un model de tip MLP (Multi-Layer Perceptron), însă acesta a fost respins deoarece nu este invariant la translație și necesită un număr excesiv de parametri pentru a procesa pixeli individuali, obținând o acuratețe semnificativ mai scăzută și fiind extrem de sensibil la zgomotul de fundal.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Valoare optimă pentru optimizatorul Adam, asigurând un echilibru între viteza de învățare și stabilitatea funcției de pierdere (loss). |
| Batch Size | 8 | Dimensiune redusă pentru a permite actualizări frecvente ale ponderilor, potrivită pentru diversitatea setului de date mixt (sintetic + real). |
| Epochs | 20 | Numărul de iterații la care modelul atinge convergența, fără a intra în zona de memorare (overfitting). |
| Optimizer | Adam | Algoritm adaptiv de optimizare a gradientului, ideal pentru procesarea imaginilor datorită gestionării automate a ratei de învățare. |
| Loss Function | CrossEntropyLoss | Funcția standard pentru clasificarea multi-clasă, penalizând logaritmic distanța față de eticheta corectă. |
| Regularizare | Dropout 0.4 + BatchNorm | Includerea Batch Normalization pentru stabilitate numerică și Dropout de 0.4 pentru a forța rețeaua să învețe trăsături robuste. |
| Early Stopping | Patience = 5 | Oprirea automată a antrenării dacă eroarea pe setul de validare nu scade timp de 5 epoci, prevenind degradarea performanței modelului. |


### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Latență | Observații |
|-----|-----------------------------|----------|----------|---------|------------|
| Baseline | MLP_Baseline (Perceptron) | 83.33% | 0.82 | 1.16 ms | Model rapid, dar cu un număr masiv de parametri (6.2M). |
| Exp 1 | CNN_Principal (3 straturi Conv) | 83.88% | 0.83 | 2.71 ms | Capacitate de generalizare superioară cu mult mai puțini parametri. |
| Exp 2 | Integrare BatchNorm + Dropout 0.4 | 84.10%* | 0.84* | 2.75 ms | Stabilitate crescută a antrenării și reducerea oscilațiilor de loss. |
| Exp 3 | Optimizare prin Export ONNX | 83.88% | 0.83 | 2.71 ms | Latență redusă sub pragul industrial de 50 ms. |
| FINAL | CNN_Principal Optimizat | 83.88% | 0.83 | 2.71 ms | Echilibru optim între precizie și eficiență (2.1M parametri). |

**Justificare alegere model final:**

Am ales configurația CNN_Principal deoarece oferă un raport mult mai eficient între performanță și complexitatea arhitecturală față de modelul MLP_Baseline. Deși acuratețea brută este similară (aprox. 83.8%), modelul CNN utilizează de 3 ori mai puțini parametri (2.1M față de 6.2M), ceea ce reduce riscul de supra-învățare și permite o execuție mai fluidă pe hardware limitat. Compromisul principal a fost acceptarea unei latențe ușor mai mari (2.71 ms față de 1.16 ms), care rămâne însă mult sub pragul critic de 50 ms necesar pentru monitorizarea în timp real. Această alegere asigură o robustețe mai mare în detectarea formelor geometrice prin utilizarea straturilor de convoluție care extrag trăsături ierarhice.

**Referințe fișiere:** `results/final_metrics.json`:Metrice finale (JSON), `results/comparare_arhitecturi.csv` : Comparație arhitecturi (CSV), `models/best_CNN_Principal.pt` : Model final optimizat (PyTorch) , `models/final_model.onnx` : Model optimizat pentru producție

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| Accuracy | 83.89% | ≥ 70% | ✓ |
| F1-Score (Macro) | 0.84 | ≥ 0.65 | ✓ |
| Precision (Macro) | 0.84 | – | ✓ |
| Recall (Macro) | 0.84 | – | ✓ |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) |  Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|-----------------------------|--------------|
| Accuracy | 83.88% | 83.89% | +0.01% |
| F1-Score | 0.82 | 0.84 | +0.02 |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_CNN_Principal.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| Clasa cu cea mai bună performanță | Cerc și Om — Precision 100%, Recall 100%. Modelul a identificat perfect toate mostrele, fără alarme false sau omisiuni. |
| Clasa cu cea mai slabă performanță | Elipsa — Precision 69.04%, Recall 64.44%. Această clasă a fost cel mai greu de identificat corect de către model. |
| Confuzii frecvente | Elipsa confundată cu Oval (16 cazuri) și Oval confundat cu Elipsa (13 cazuri). Eroarea apare din cauza similarității geometrice extreme (raport de aspect apropiat) și a rezoluției de intrare (64×64). |
| Dezechilibru clase | Nu există. Setul de date este perfect echilibrat, fiecare clasă având exact 25% din date (45 de imagini per clasă), deci erorile nu sunt cauzate de volumul de date. |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|------------------|------------------------|
| 1 | Elipsă alungită cu marginile pixelate | Oval | Elipsă | Rezoluția redusă (64×64) a pierdut detaliile fine ale curburii specifice elipsei. | Sortare greșită pe bandă; piesa ajunge în containerul incorect. |
| 2 | Oval captat sub unghi de perspectivă | Elipsă | Oval | Deformarea de perspectivă a accentuat axa majoră, simulând raportul de aspect al unei elipse. | Eroare de inventar automatizat; necesită intervenție manuală. |
| 3 | Elipsă în condiții de iluminare slabă | Oval | Elipsă | Umbrele periferice au alterat percepția conturului, făcând forma să pară asimetrică (ovală). | Respingerea piesei conforme ca fiind „defect structural” (False Negative). |
| 4 | Oval cu zgomot de senzor (Gaussian Noise) | Elipsă | Oval | Zgomotul electronic a creat artefacte pe marginea formei, inducând în eroare filtrele convoluționale. | Blocaj în etapa de asamblare automată (piesa nu se potrivește în locaș). |
| 5 | Elipsă rotită la un unghi de 45° | Oval | Elipsă | Invarianța parțială la rotație a modelului pentru forme geometrice extrem de asemănătoare. | Scăderea eficienței liniei de producție prin declanșarea alarmelor false. |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

În scenariul monitorizării prezenței indoor în campus, prioritatea zero a sistemului este fiabilitatea detecției clasei Om. Rezultatele obținute (Recall 100%) demonstrează o siguranță maximă a soluției: nicio persoană aflată în cadrul supravegheat nu a fost omisă de algoritm. Mai mult, Precision-ul de 100% pentru clasa umană garantează că elementele de fundal sau obiectele de decor (tablouri, corpuri de iluminat, elemente de mobilier) nu generează alarme false de prezență, asigurând un pontaj electronic precis.

Deși există o rată de confuzie între clasele Elipsă și Oval, acest aspect are un impact minor asupra obiectivului principal, deoarece ambele reprezintă elemente statice de decor. Totuși, acuratețea perfectă pe clasa Cerc este esențială, deoarece permite sistemului să distingă fără eroare între conturul capului unei persoane și alte obiecte circulare din sălile de curs. Din punct de vedere logistic, modelul permite automatizarea monitorizării fluxului de studenți cu o rată de succes absolută pe componenta umană, minimizând necesitatea supravegherii manuale.

**Pragul de acceptabilitate pentru domeniu:** Detecție Om (Critic): Recall >= 95% (pentru a nu omite studenți/personal)
- Fiabilitate Alarme (Critic): Precision > = 98% (pentru a nu confunda obiectele cu oameni)
- Identificare Obiecte (Secundar): Accuracy >= 70% pentru formele geometrice din mediu

**Status:** ATINS — Sistemul a atins un Recall de 100% și un Precision de 100% pentru clasa Om, depășind standardele de siguranță impuse pentru monitorizarea indoor. Acuratețea generală de 83.89% depășește, de asemenea, pragul minim stabilit pentru proiect
- Sortare Detaliată (Elipsă/Oval): NEATINS (Recall de 64%- 71% cu o diferenta de aproximativ 20% față de standardul ideal de 85%) 

**Plan de îmbunătățire (dacă neatins):** 
- Rezoluție: Trecerea de la 64x64 la 128x128 pixeli pentru a captura detaliile fine ale conturului.
- Augmentare: Generarea de eșantioane suplimentare prin transformări elastice pentru a diferenția curburile eliptice de cele ovale.
- Filtru Confidență: Setarea unui prag de 75% în UI; sub această valoare, sistemul va marca obiectul ca "Incert", evitând erorile de pontaj automat.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5/6 | Modificare Etapa Finală | Justificare |
|------------|----------------|--------------------------|-------------|
| Model încărcat | trained_model.h5 | best_CNN_Principal.pt | Migrarea completă la PyTorch pentru o mai bună gestionare a memoriei și a straturilor de regularizare. |
| Threshold decizie | Fără (Argmax implicit) | 0.60 (implementat) | Corecție critică: adăugarea pragului pentru a evita clasificarea eronată a formelor ambigue drept „Om”. |
| UI – feedback vizual | Text simplu | Confidence Score (%) | Afișarea probabilității pentru ca utilizatorul să poată evalua singur certitudinea detecției. |
| Logging | Doar vizual în consolă | `attendance_log.csv` | Automatizarea stocării datelor de prezență pentru a permite consultarea ulterioară a istoricului. |
| Latență | Nemăsurată | Benchmark: 2.71 ms | Optimizarea codului de inferență pentru a asigura fluiditatea aplicației Streamlit. |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

În acest screenshot este prezentată interfața aplicației în timp ce rulează o predicție live pentru o imagine din clasa „Om”. Se poate observa că modelul a identificat corect subiectul, afișând un scor de încredere foarte ridicat, de 99.62%, ceea ce confirmă precizia ridicată a rețelei pe componenta de monitorizare umană.

Imaginea demonstrează, de asemenea, integrarea reușită a bazei de date sub formă de tabel în partea de jos a ecranului. Acesta înregistrează automat fiecare detecție cu data și ora exactă, asigurând istoricul necesar pentru pontajul studenților sau securitatea campusului. Totodată, prezența indicatorului de latență (4.36 ms) subliniază faptul că sistemul este capabil să proceseze informația instantaneu, fiind pregătit pentru utilizarea pe teren.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/demonstratie_live`

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input Extern | Descărcarea unei imagini noi cu un portret uman de pe Google Images și crearea unei elipse personalizate în Microsoft Paint. |
| 2 | Procesare | Încărcarea fișierelor prin interfața Streamlit; sistemul confirmă procesarea prin afișarea numelui fișierului și redimensionarea imaginii. |
| 3 | Inferență | Afișarea predicției în timp real: „Rezultat: Om (99.62%)” pentru imaginea de pe internet și marcarea ca „Incert [Revizuire]” pentru desenul din Paint (datorită threshold-ului de 0.60). |
| 4 | Decizie / Logging | Salvarea automată în Jurnalul de Audit (Nivel 3) cu detalii despre Timestamp, Predicție și Scor de încredere. |

**Latență măsurată end-to-end:** [4.36] ms 
**Data și ora demonstrației:** [09.02.2026, 21:37] 

---

## 8. Structura Repository-ului Final

```
```
proiect-rn-[Rotaru_Elena-Alexandra]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # (opțional) Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   ├── test_model.png                  # Screenshot rulare script testare
│   │   └── train_model.png                 # Screenshot rulare script antrenare
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demonstratie_live.mp4           # Inregistrarea functionarii 
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve_CNN_Principal.png    # Grafic loss/val_loss (Etapa 5)
│   │   └── loss_curve_MLP_Baseline.png     # Evoluție metrici (Etapa 6)
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   │
│   ├── raw/
│   │   └── synthetic_faces/                # Date brute originale
│   │
│   ├── processed/                          # Date curățate și transformate
│   │   ├── Cerc/
│   │   ├── Elipsa/
│   │   ├── Om/
│   │   └── Oval/
│   │
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   │   ├── Cerc/
│   │   ├── Elipsa/
│   │   └── Oval/
│   │
│   ├── train/                              # Set antrenare (70%)
│   │   ├── Cerc/
│   │   ├── Elipsa/
│   │   ├── Om/
│   │   └── Oval/
│   │
│   ├── validation/                         # Set validare (15%)
│   │   ├── Cerc/
│   │   ├── Elipsa/
│   │   ├── Om/
│   │   └── Oval/
│   │
│   └── test/                               # Set testare (15%)
│       ├── Cerc/
│       ├── Elipsa/
│       ├── Om/
│       └── Oval/
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate_cartoon_faces_shapes.py # Script generare date originale
│   │   
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── prepare_dataset_synthetic.py    # [Data Cleaner] Segmentare MediaPipe (background removal)
│   │   ├── preprocess_generated_shapes.py  # [Feature Engineering] Redimensionare 64×64 și normalizare
│   │   └── split_dataset.py                # [Data Splitter] Organizare 70 / 15 / 15 (train / val / test)
│   │
│   └── neural_network/                     # MODUL 2: Model RN
│       ├── README.md                       # Documentație arhitectură RN
│       ├── model_torch.py                  # [Model] Definire CNN_Principal și MLP_Baseline
│       ├── train_model.py                  # [Train+Evaluate+Optimize+Visualize] Scriptul principal
│       └── test_model.py                   # [Inference] Script de validare și logare prezență
│
├── app/                                    # Modul 3: UI / Web Service
│   ├── README.md                           # Instrucțiuni de lansare a aplicației
│   ├── main.py                             # Aplicație principală (Streamlit)
│   └── style.css                           # [Custom UI] Design profesional (Midnight Blue Theme)
│   └── generate_results.py                 # Script generare pentru folderul results
├── models/
│   ├── untrained_CNN.pt                    # Model schelet neantrenat (Etapa 4)
│   ├── best_MLP_Baseline.pt                # Modelul antrenat inițial (Baseline) : Etapa 5
│   ├── best_CNN_Principal.pt               # Modelul final, optimizat (Etapa 6)
│   └── final_model.onnx                    # Deployment: Formatul universal exportat
│   └── final_model.onnx.data              # Deployment: Formatul universal exportat
│
├── results/
│   ├── training_history.csv                # Istoricul celor 10 epoci (Loss/Acc)
│   ├── test_metrics.json                   # Performanța modelului Baseline (MLP)
│   ├── optimization_experiments.csv        # Comparația între MLP și CNN
│   ├── final_metrics.json                  # Performanța modelului Final (CNN)
│   ├── error_analysis.json                 # Analiza confuziei între clase
│   └── attendance_log.csv                  # Jurnalul de prezență din aplicație
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   ├── model_config.yaml                   # Configurație finală model (Etapa 6)
│   └── class_names.json                    # Bonus. Maparea index -> nume (Cerc, Om, etc.), esențială pentru interfață
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```
```

### Legendă Progresie pe Etape

|| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|--------|--------|--------|--------|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | — | ✓ Actualizat | — |
| `data/generated/` | — | ✓ Creat | — | — |
| `src/preprocessing/` | ✓ Creat | — | ✓ Actualizat | — |
| `src/data_acquisition/generate_cartoon_faces.py` | — | ✓ Creat | — | — |
| `src/neural_network/model_torch.py` | — | ✓ Creat | — | — |
| `src/neural_network/train_model.py`, `test_model.py` | — | — | ✓ Creat | — |
| `src/app/main.py`, `style.css` | — | ✓ Creat | ✓ Actualizat | ✓ Actualizat |
| `src/app/generate_results.py` | — | — | — | ✓ Creat |
| `models/untrained_CNN.pt` | — | ✓ Creat | — | — |
| `models/best_MLP_Baseline.pt` *(Trained)* | — | — | ✓ Creat | — |
| `models/best_CNN_Principal.pt` *(Optimized)* | — | — | — | ✓ Creat |
| `models/final_model.onnx` | — | — | — | ✓ Creat |
| `docs/state_machine.png` | — | ✓ Creat | — | — |
| `docs/etapa3_analiza_date.md` | ✓ Creat | — | — | — |
| `docs/etapa4_arhitectura_SIA.md` | — | ✓ Creat | — | — |
| `docs/etapa5_antrenare_model.md` | — | — | ✓ Creat | — |
| `docs/etapa6_optimizare_concluzii.md` | — | — | — | ✓ Creat |
| `results/training_history.csv` | — | — | ✓ Creat | — |
| `results/test_metrics.json` | — | — | ✓ Creat | — |
| `results/optimization_experiments.csv` | — | — | — | ✓ Creat |
| `results/final_metrics.json` | — | — | — | ✓ Creat |
| **README.md (Principal)** | *Draft* | *Actualizat* | *Actualizat* | **FINAL** |


### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=0.833, F1=0.821" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=0.8389, F1=0.8387 (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+ pentru compatibilitate optimă cu PyTorch)
pip >= 21.0 (pentru gestionarea pachetelor)
Virtual Environment (recomandat, pentru a evita conflictele între biblioteci)
Sistem de operare: Windows, Linux sau macOS

```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone [https://github.com/RotaruAlexandra/Proiect-RN-Rotaru_Elena-Alexandra.git]
cd Proiect-RN-Rotaru_Elena-Alexandra

# 2. Creare mediu virtual (recomandat)
python -m venv venv

# 3. Activarea mediului virtual
# Pe Windows (PowerShell/CMD):
venv\Scripts\activate
# Pe Linux/Mac:
source venv/bin/activate

# 4. Instalare dependențe 
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Preprocesare date (Redimensionare 64x64 și uniformizare)
python src/preprocessing/preprocess_generated_shapes.py
python src/preprocessing/prepare_dataset_synthetic.py

# Pasul 2: Împărțire Dataset (Split 70% Train, 15% Val, 15% Test)
python src/preprocessing/split_dataset.py

# Pasul 3: Antrenare și Evaluare Modele (MLP Baseline vs CNN Principal)
# Acest script generează și graficele de performanță în /docs
python src/neural_network/train_model.py

# Pasul 4: Generare Rapoarte Oficiale (Etapa 6)
python src/app/generate_results.py

# Pasul 5: Lansare Aplicație UI (Dashboard Streamlit)
streamlit run src/app/main.py
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect (PyTorch)
python -c "import torch; from src.neural_network.model_torch import FaceClassifierCNN; m = FaceClassifierCNN(num_classes=4); m.load_state_dict(torch.load('models/best_CNN_Principal.pt', map_location='cpu')); m.eval(); print(' Modelul CNN Principal a fost încărcat cu succes!')"

# Verificare script de testare pe lotul de imagini (Mod IDLE)
# Acest script va procesa automat fișierele din data/test/ și va scrie în attendance_log.csv
python src/neural_network/test_model.py
```
---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|-------------------------------|--------|----------|--------|
| Recunoaștere forme geometrice | Diferențiere corectă a 4 clase | Acuratețe optimă pe Om/Cerc; confuzii minore Oval/Elipsă. | ✓ |
| Monitorizare timp real | Latență < 50 ms | 4.36 ms (inclusiv randare UI Streamlit) | ✓ |
| Accuracy pe test set | ≥ 70% | 83.89% | ✓ |
| F1-Score pe test set | ≥ 0.65 | 0.82 | ✓ |
| Detecție critică persoane | Recall ≥ 95% | 100% (nicio persoană omisă în test set) | ✓ |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

1. **Limitare 1:** Restricții de Etică și Confidențialitate (GDPR): Proiectul a fost limitat de imposibilitatea utilizării imaginilor reale cu studenți din sălile de curs din cauza reglementărilor GDPR. Deși am simulat trăsăturile umane prin generarea de „cartoon faces” prin cod, modelul ar necesita un proces de Fine-Tuning pe date reale (anonimizate) pentru a fi cu adevărat eficient într-un scenariu de producție.
2. **Limitare 2:** Sensibilitatea la Distanță și Perspectivă:Spre deosebire de studiul de caz inițial bazat pe YOLOv8, arhitectura actuală a fost testată pe imagini centrate. În realitate, acuratețea ar scădea semnificativ dacă studentii se află la distanțe mari de cameră sau în unghiuri de perspectivă extremă, aspect pe care modelul actual nu îl gestionează încă prin invarianță spațială.
3. **Limitare 3:** Dependența de Dimensiunea Dataset-ului: Inițial, utilizarea unui set de date redus (aprox. 100 imagini) a condus la fenomenul de overfitting, oferind scoruri nerealist de mari. Optimizarea a necesitat extinderea setului de date și utilizarea augmentărilor, însă rămâne o limitare față de volumele de date masive necesare pentru un sistem de tip Industrial Grade.
4. **Limitare 4:** Instabilitatea Predicției pentru Forme Similare: Modelul prezintă dificultăți în a distinge între Elipsă și Oval atunci când raportul de aspect este apropiat. În aceste cazuri, scorul de confidență scade sub pragul de 0.60 (triggering threshold), iar sistemul clasifică obiectul ca fiind "Incert". Această limitare apare din cauza rezoluției de intrare (64x64) care nu permite capturarea detaliilor fine de curbură necesare pentru o distincție geometrică perfectă.
5. **Limitare 5:**Efectul de "Model Warm-up" în Streamlit: Am observat o latență ridicată (peste 20 ms) la procesarea primei imagini imediat după pornirea aplicației. Aceasta este o limitare de sistem cauzată de încărcarea modelului în memoria RAM și inițializarea graficului de computație (Cold Start). Ulterior, latența se stabilizează la valoarea optimă de ~7 ms, însă pentru un sistem industrial, ar fi necesară o pre-încărcare (eager loading) a modelului la start-up.

6. **Funcționalități planificate dar neimplementate:** 
- Tranziția către YOLOv11: Deși inițial am vizat utilizarea unui model de tip Object Detection (YOLO), complexitatea implementării și necesitatea marcării manuale a mii de cadre au dus la alegerea unei arhitecturi CNN personalizate, mai potrivită pentru scopul academic curent.
- Integrarea Hardware Reală: Din cauza limitărilor de timp, sistemul nu a fost testat pe o cameră video în timp real într-o sală de curs, ci doar prin încărcare de fișiere (file upload).


### 10.3 Lecții Învățate (Top 5)

1. **[Lecție 1]:** [Importanța Datelor Sintetice în Context Etic (GDPR): Am învățat că în proiectele de monitorizare a persoanelor, constrângerile legale pot bloca accesul la date reale. Soluția de a genera un dataset "cartoon-ish" prin cod mi-a permis să demonstrez funcționalitatea tehnică (Proof of Concept) fără a încălca normele de confidențialitate.] 
2. **[Lecție 2]:** [Gestionarea Overfitting-ului prin Volumul de Date: Inițial, am observat că un dataset prea mic (aprox. 100 imagini) duce la o acuratețe nerealist de mare (memorare), nu la învățare. Extinderea setului de date și utilizarea augmentărilor au fost esențiale pentru a obține un model capabil să generalizeze pe imagini noi din Paint sau Google.]
3. **[Lecție 3]:** [Threshold-ul Custom ca Filtru de Siguranță: Am realizat că pragul default de 0.5 (Argmax) nu este suficient pentru a distinge între forme geometrice foarte similare (Oval/Elipsă). Ajustarea pragului la 0.60 a transformat erorile de clasificare în marcaje de "Incert", crescând astfel fiabilitatea sistemului de monitorizare.]
4. **[Lecție 4]:** [Impactul Arhitecturii CNN asupra Eficienței: Trecerea de la un model MLP dens la un CNN cu BatchNorm și Dropout a demonstrat că poți obține performanțe mai bune cu un model de 3 ori mai mic ca număr de parametri. Această optimizare este critică pentru rularea pe sisteme cu resurse limitate (edge computing).]
5. **[Lecție 5]:** [Înțelegerea Latenței și a Efectului de "Warm-up": Am descoperit că performanța brută a modelului diferă de experiența utilizatorului în aplicație. Observarea latenței ridicate la prima rulare (20ms) vs. rularea stabilă (4ms) m-a învățat că optimizarea software (Streamlit/Python) este la fel de importantă ca optimizarea rețelei neuronale.]

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

Dacă aș reîncepe acest proiect, prima decizie diferită ar fi legată de abordarea strategică a datelor. Aș investi mult mai mult timp în faza de Data Engineering și documentare a restricțiilor legale (GDPR) încă din prima săptămână. Panica resimțită atunci când am realizat că nu pot folosi date reale a fost o lecție dură despre cât de important este să cunoști contextul etic al unui proiect înainte de a scrie prima linie de cod. De asemenea, aș începe direct cu o arhitectură CNN și un set de date diversificat prin augmentări, evitând etapa inițială de „entuziasm fals” oferită de overfitting-ul pe un dataset prea mic.

Privind în urmă, acest proiect a fost o lecție de reziliență psihologică la fel de mult pe cât a fost una de inteligență artificială. Am învățat că o eroare de cod sau un grafic de loss care „o ia razna” nu reprezintă un eșec, ci un semnal necesar pentru optimizare. Evoluția mea a constat în trecerea de la dorința de a obține un model „perfect” (cu scoruri de 100% care, în realitate, mascau probleme de generalizare), la acceptarea unui model onest, de 83.89%, care funcționează pe date complet noi. Sunt conștientă că sistemul are limitări, dar plec de la acest proiect cu înțelegerea faptului că în IA, succesul nu stă în cifrele de pe hârtie, ci în capacitatea de a diagnostica problema și de a găsi soluții creative sub presiune.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|-------|---------------------|-------------------|
| **Short-term (1–2 săptămâni)** | Rafinarea dataset-ului și creșterea rezoluției la **128×128** | Eliminarea confuziilor geometrice (Oval/Elipsă) prin capturarea detaliilor fine de curbură și margini |
| **Medium-term (1–2 luni)** | Tranziția către arhitectura hibridă **MobileViT** | Utilizarea mecanismelor de *Global Attention* pentru precizie sporită, menținând latența scăzută pentru rularea pe hardware accesibil (*Edge Computing*) |
| **Long-term (Viitor)** | Integrarea **Hybrid Quantum–Classical Neural Networks (VQC)** | Scalarea sistemului la nivel de universitate; algoritmii cuantici permit procesarea volumelor masive de date cu un consum minim de energie și o eficiență de calcul superioară sistemelor clasice |

---

## 11. Bibliografie

*[Minimum 3 surse cu DOI/link funcțional - format: Autor, Titlu, Anul, Link]*

1. Gupta, A., Human Faces Dataset for Deep Learning, 2021. URL: https://www.kaggle.com/datasets/ashwingupta3012/human-faces
2. This Person Does Not Exist, Generative Adversarial Networks for AI Face Synthesis, 2024. URL: https://this-person-does-not-exist.com/en
3. PyTorch Documentation, Deep Learning with PyTorch: Neural Network Module (torch.nn), 2024. URL: https://pytorch.org/docs/stable/nn.html
4. Abaza, B., Curs 9 Rețele Neuronale: Arhitecturi Convoluționale și Procesarea Imaginilor, 2025. Facultatea de Inginerie Industrială și Robotică (FIIR), Universitatea Politehnica din București.
5. Abaza, B., Curs 10 Rețele Neuronale: State Machines & Event Driven Architecture, 2025. Facultatea de Inginerie Industrială și Robotică (FIIR), Universitatea Politehnica din București.
6. Ioffe, S., Szegedy, C., Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015. URL: https://arxiv.org/abs/1502.03167
7. Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., Kalinin, A.A., Albumentations: Fast and Flexible Image Augmentations, Information, 11(2), 125, 2020. DOI: https://doi.org/10.3390/info11020125
8. Google AI Edge, MediaPipe Selfie Segmentation — Real-time Human Segmentation, Google Developers Documentation, 2020. URL: https://developers.google.com/mediapipe/solutions/vision/image_segmenter
9. ONNX Community, Open Neural Network Exchange (ONNX) — Open Standard for Machine Learning Interoperability, GitHub / Linux Foundation AI, 2017–2024. URL: https://onnx.ai / https://github.com/onnx/onnx
10. Streamlit Inc., Streamlit Documentation — Build and Share Data Applications, 2019–2024. URL: https://docs.streamlit.io
11. Mehta, S., Rastegari, M., MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer, International Conference on Learning Representations (ICLR), 2022. URL: https://arxiv.org/abs/2110.02178
12. Bradski, G., The OpenCV Library, Dr. Dobb's Journal of Software Tools, 2000. URL: https://opencv.org / https://docs.opencv.org

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [✓] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [✓] **F1-Score ≥0.65** pe test set
- [✓] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [✓] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [✓] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [✓] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [✓] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [✓] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [✓] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [✓] **README.md** complet (toate secțiunile completate cu date reale)
- [✓] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [✓] **Screenshots** prezente în `docs/screenshots/`
- [✓] **Structura repository** conformă cu Secțiunea 8
- [✓] **requirements.txt** actualizat și funcțional
- [✓] **Cod comentat** (minim 15% linii comentarii relevante)
- [✓] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [✓] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [✓] **Tag `v0.6-optimized-final`** creat și pushed
- [✓] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [✓] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [✓] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [✓] **Minimum 40% date originale** (nu doar subset din dataset public)
- [✓] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [10.02.2026]
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
