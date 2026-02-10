# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Rotaru Elena-Alexandra 
**Link Repository GitHub:** https://github.com/RotaruAlexandra/Proiect-RN.git
**Data predării:** 11.12.2025- 18.12.2025

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Această etapă are ca scop antrenarea efectivă a modelului de rețea neuronală definit în Etapa 4, evaluarea performanței acestuia și validarea funcționării reale prin inferență.Modelul a fost antrenat folosind PyTorch, pe un dataset combinat care include ≥40% date originale generate.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

  
### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

1. Antrenare model
Modelul a fost antrenat pe un set de date diversificat, conținând formele geometrice (Cerc, Elipsa, Om, Oval), cu peste 40% imagini originale colectate manual.

2. Configurație
S-au efectuat 10 epoci de antrenare cu un batch size de 8, asigurând o actualizare frecventă a ponderilor pentru o învățare detaliată.

4. Împărțire stratificată
Datele au fost distribuite conform cerințelor: 70% pentru antrenare (Train), 15% pentru validare (Val) și 15% pentru testarea finală (Test).

## Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare optimă pentru algoritmul Adam; permite pași de învățare suficienți pentru a scădea loss-ul sub 0.4 rapid. |
| Batch size | 8 | Potrivit pentru un set de date de dimensiuni medii, asigurând mai multe actualizări ale gradientului per epocă. |
| Number of epochs | 10 | Aleasă pe baza curbei de loss, care arată că după epoca 6 modelul se stabilizează și eroarea nu mai scade semnificativ. |
| Optimizer | Adam | Algoritm adaptiv care ajustează rata de învățare pentru fiecare parametru, ideal pentru arhitecturi CNN. |
| Loss function | CrossEntropyLoss | Funcția standard pentru probleme de clasificare multi-class, penalizând erorile de tip Elipsă vs Oval. |
| Activation functions | ReLU (hidden), Softmax (output) | ReLU elimină saturația gradientului, iar Softmax transformă ieșirile în probabilități interpretate de UI (ex. 99.86%). |

**Justificare detaliată batch size:**

Am ales batch_size=8 deoarece am un set de date de dimensiuni moderate. 
Această valoare asigură:
- O granularitate mai mare a învățării, oferind modelului mai multe oportunități de a-și ajusta 
  ponderile în cadrul aceleiași epoci.
- Adaptabilitate pe hardware cu resurse limitate (CPU), menținând un consum redus de memorie.
- Convergență stabilă, după cum se observă în Loss Curve unde eroarea scade rapid după primele epoci.
  
5. Metrici calculate pe test set
   
-Acuratețe: 83.88% (151 predicții corecte din 180 de eșantioane de test).
-F1-score (macro): ~0.82, indicând o performanță solidă, deși afectată de confuziile între formele geometrice similare.

6. Salvare model antrenat
Modelul optim a fost salvat în folderul de proiect la calea: models/best_CNN_Principal.pt.

7. Integrare în UI
   
-Interfața Streamlit încarcă dinamic modelul .pt salvat anterior.
-Sistemul realizează inferență în timp real pe imagini noi, afișând clasa și scorul de încredere (ex: 99.86% pentru Cerc).
-Screenshot demonstrativ disponibil în docs/screenshots/inference_real.png.

### Nivel 2 – Recomandat (85-90% din punctaj)

1. Early Stopping
Sistemul monitorizează constant eroarea pe setul de validare (val_loss). Antrenarea se oprește automat dacă val_loss nu scade timp de 5 epoci consecutive pentru a preveni supra-antrenarea, asigurând astfel că modelul păstrează cele mai bune ponderi (best_weights).

2. Learning Rate Scheduler
Am utilizat ReduceLROnPlateau pentru a ajusta rata de învățare. Atunci când eroarea stagnează, scheduler-ul reduce automat learning rate-ul (de exemplu, de la 0.001 la 0.0001), permițând modelului să realizeze ajustări mai fine ale greutăților pentru a găsi minimul global.

3. Augmentări relevante domeniu
   
Pentru a crește robustețea sistemului în contextul formelor geometrice și industriale, am aplicat următoarele transformări:
-Lighting Variation: Ajustarea luminozității și a contrastului pentru a simula condiții de iluminare variabile dintr-o hală industrială.
-Slight Perspective: Deformări ușoare de perspectivă pentru a simula unghiuri diferite de captură ale camerei fixe față de obiectele de pe banda de producție.
-Gaussian Noise: Adăugarea unui zgomot controlat pentru a simula interferențele electronice ale senzorului camerei.

4. Grafic loss și val_loss
   
-Graficul arată o convergență stabilă, eroarea scăzând abrupt în primele 2 epoci și stabilizându-se ulterior sub pragul de 0.4 pentru setul de validare.
-Fișier salvat în: docs/loss_curve_CNN_Principal.png.

5. Analiză erori context industrial (OBLIGATORIU Nivel 2)
   
Pe baza matricei de confuzie obținute, am identificat următoarele aspecte critice:
-Performanță Maximă: Clasele Cerc și Om au o acuratețe de 100% (45/45 detecții corecte), ceea ce indică faptul că modelul a extras trăsături foarte clare pentru aceste obiecte.
-Confuzie Critică (Elipsa vs Oval):
->16 Elipse au fost clasificate greșit ca Ovale.
->13 Ovale au fost clasificate greșit ca Elipse.
-Impact Industrial: În contextul unei benzi de sortare, această eroare ar putea duce la trimiterea unui produs oval în containerul de elipse. Cauza este similitudinea geometrică ridicată (raportul de aspect similar), modelul având nevoie de o rezoluție mai mare sau de mai multe epoci de antrenare pentru a distinge aceste diferențe fine.

-Indicatori țintă Nivel 2 realizați:
->Acuratețe: 83.88% (Peste pragul de 75%).
->F1-score (macro): 0.82 (Peste pragul de 0.70).

### Nivel 3 – Bonus (până la 100%)

1. Comparare 2+ arhitecturi diferite
Pentru a identifica cea mai eficientă soluție de monitorizare, am antrenat și comparat arhitectura propusă (CNN_Principal) cu un model de referință de tip perceptron multistrat (MLP_Baseline).

## Comparare Arhitecturi (Nivel 3 Bonus)

| **Arhitectură** | **Acuratețe Test** | **F1-Score (Macro)** | **Latență (ms)** | **Nr. Parametri** |
|-----------------|-------------------|---------------------|------------------|-------------------|
| **CNN_Principal**  | **83.89%** | **0.838** | 2.71 ms | 2,192,132 |
| MLP_Baseline | 83.33% | 0.820 | 1.16 ms | 6,294,020 |

Justificare alegere finală: Deși ambele modele au obținut acuratețe similară, am ales CNN_Principal ca model final. Această arhitectură oferă un F1-Score superior (0.838), demonstrând o capacitate mai bună de generalizare pe toate clasele, utilizând în același timp de 3 ori mai puțini parametri decât modelul MLP.

2. Export ONNX + Benchmark latență
Modelul a fost optimizat pentru execuție industrială prin exportul în format ONNX.

Fișier: models/final_model.onnx

Benchmark latență: Testele de performanță indică o latență de procesare de doar 2.71 ms per imagine. Această valoare este mult sub pragul critic de 50ms, permițând monitorizarea în timp real a liniilor de producție de mare viteză.

3. Confusion Matrix + Analiză erori (5 Exemple/Cauze)
Analiza matricei de confuzie pentru CNN_Principal evidențiază succesul total pe clasele Cerc și Om (45/45 predicții corecte), dar și dificultăți specifice:

Analiza eșecurilor principale:
-Elipsă → Oval (16 cazuri): Cea mai frecventă eroare; cauzată de similitudinea geometrică ridicată (ambele fiind figuri curbe alungite).
-Oval → Elipsă (13 cazuri): Confuzie simetrică ce apare când un obiect oval este captat sub un unghi care îi accentuează axa majoră.
-Scor încredere scăzut (ex: 55.24%): Observat în dashboard-ul live pentru clasa 'Elipsa', indicând ambiguitate în extragerea trăsăturilor de contur.
-Zgomot de senzor: În context industrial, zgomotul electronic al camerei poate altera pixelii de margine, făcând o elipsă să pară mai "rugoasă", similară cu un oval.
-Rezoluție limitată: Procesarea la 64x64 pixeli elimină detaliile fine ale raportului de aspect care diferențiază elipsele de ovale.

Implicații și Măsuri Corective

Implicații industriale:
-False Positives (Elipsă clasificată ca Oval): Ar putea duce la sortarea greșită a pieselor, riscând blocaje în etapele de asamblare automată.
-Prioritate: Reducerea confuziei între elipse și ovale pentru a atinge o precizie de peste 90% pe aceste clase dificile.

Măsuri corective propuse:
1.Augmentare perspectivă: Utilizarea transformărilor de tip PerspectiveTransform pentru a învăța modelul să recunoască formele din orice unghi de incidență al camerei.
2.Loss ponderat: Aplicarea unor penalizări mai mari pentru confuziile între clasele Elipsă și Oval în timpul antrenării.
3.Arhitectură reziduală: Trecerea la un model de tip ResNet pentru a extrage trăsături mai profunde necesare diferențierii geometrice fine.

## Structura Repository-ului la Finalul Etapei 5
**Clarificare organizare:**
proiect-rn-Rotaru-Alexandra/
├──  config/                       # Configurații sistem
│   ├── class_names.json            # Mapare indici -> Nume clase (Cerc, Elipsa, Om, Oval)
│   ├── model_config.yaml           # Detalii arhitectură model
│   └── preprocessing_params.pkl    # Parametri normalizare și scalare (Etapa 3)
│
├── data/                         # Seturi de date (Stratificare 70/15/15)
│   ├── raw/                        # Imagini brute (originale + sintetice)
│   ├── processed/                  # Imagini după preprocesare
│   ├── train/                      # Set de antrenare
│   ├── validation/                 # Set de validare
│   └── test/                       # Set de testare (evaluare finală)
│
├──  docs/                         # Documentație și diagrame
│   ├── datasets/                   # Rapoartele etapelor anterioare (E3, E4, E5)
│   │   └── state_machine.png       # Diagrama de stări a sistemului
│   └── screenshots/                # Capturi de ecran cu interfața și inferența
│       └── inference_real.png      # Demonstrație Nivel 3 - Inferență Live
│
├──  models/                       # Modelele salvate
│   ├── best_CNN_Principal.pt       # Modelul principal (PyTorch)
│   ├── best_MLP_Baseline.pt        # Modelul de referință pentru comparare
│   └── final_model.onnx            # Export industrial (Nivel 3 Bonus)
│
├── results/                      # Rezultate și loguri
│   ├── attendance_log.csv          # Jurnal audit generat în timp real
│   ├── comparare_arhitecturi.csv   # Tabel comparativ CNN vs MLP
│   └── final_metrics.json          # Metrici finale (Acuratețe, F1, Latență)
│
├──  src/                          # Codul sursă Python
│   ├── app/                        # Aplicația Streamlit (main.py)
│   └── neural_network/             # Scripturi model, train și evaluate
│
├──  Grafice Performanță (Root)
│   ├── confusion_matrix_CNN_Principal.png
│   ├── loss_curve_CNN_Principal.png
│   └── loss_curve_MLP_Baseline.png
│
├──  requirements.txt              # Dependențe proiect (torch, streamlit, etc.)
└──  README.md                     # Documentația principală


