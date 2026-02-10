# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Rotaru Elena-Alexandra 
**Link Repository GitHub** https://github.com/RotaruAlexandra/Proiect-RN.git
**Data:** 04.12.2025

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)

| Nevoie reală concretă                                          | Cum o rezolvă SIA-ul tău                                                                                       | Modul software responsabil            |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Detectarea tipului de formă facială (cartoon / synthetic face) | RN clasifică automat imaginile în una din cele 6 clase (RoundFace, OvalFace, LongFace + 3 clase GAN) în <1 sec | Modul RN (Python / LabVIEW)           |
| Necesitatea unui dataset extins fără date reale                | Generarea automată a datelor (120 cartoon faces) + curățare și preprocesare completă                           | Modul Data Generation + Preprocessing |
| Necesitatea unui tool ușor pentru utilizator                   | UI în Streamlit / LabVIEW care primește o poză și afișează clasa prezisă                                       | Modul Web Service/UI                  |


### 2. Contribuția Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

În această etapă am construit un dataset mixt format din:

imagini sintetice de fețe (50 buc.) obținute din surse publice non-GDPR

imagini complet originale generate de mine (120 buc.) prin simulare Python.

Total observații finale: 170
Observații originale: 120
Procent contribuție originală: 70.58% ✔️ (peste minimul cerut de 40%)

[X] Date generate prin simulare fizică / simulare programatică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[ ] Date sintetice prin metode avansate (GAN, FEM etc.)

Descriere detaliată a contribuției originale

Pentru a respecta cerința minimă de 40% date originale, am implementat un generator Python propriu care produce imagini pseudo-sintetice cartoonizate pe baza unor forme geometrice cu variații random. Generatorul construiește pentru fiecare exemplu o „față” în stil simplificat, compusă din:

contur de față (cerc, oval, alungit) → determină clasa

ochi, sprâncene, nas, gură, urechi

păr generat aleator

variații de poziție, culoare și proporții pentru realism

Astfel am generat 3 clase originale, fiecare conținând câte 40 de observații:

RoundFace (40 imagini)

OvalFace (40 imagini)

LongFace (40 imagini)

Toate imaginile sunt create complet de la zero, fără a folosi date externe. Fiecare imagine este unică datorită variațiilor aleatorii aplicate asupra:

poziției ochilor, nasului și gurii

grosimii trăsăturilor

culorii părului

dimensiunilor conturului feței

orientării și deformărilor mici controlate

Am folosit OpenCV pentru a genera aceste forme și pentru a introduce variații stocastice. Această metodă se încadrează în categoria „date generate prin simulare fizică / procedurale”, conform metodologiei discutate la curs.

După generare, dataset-ul original este preprocesat automat printr-un script Python dedicat, care:

redimensionează toate imaginile la 64×64 px

normalizează pixelii în intervalul 0–1

salvează datele în două formate:

JPG – pentru utilizarea ulterioară în modulul RN din LabVIEW

NPY – pentru antrenarea viitoare a rețelei în Python

În această etapă (Etapa 4), imaginile sunt doar pregătite pentru modulul RN;
antrenarea în LabVIEW nu este încă necesară și va fi realizată în Etapa 5.

Dataset-ul generat este suficient de divers vizual pentru a susține antrenarea viitoare a unei rețele neuronale multiclasă.ințelor proiectului.

**Locația codului:**  
- `src/data_acquisition/generate_cartoon_faces.py`  ← scriptul care generează cele 3 clase originale (RoundFace, OvalFace, LongFace)  
- `src/preprocessing/preprocess_generated_shapes.py` ← scriptul care redimensionează la 64×64 și normalizează imaginile generate

**Locația datelor:**  
- `data/raw/generated_shapes/`  ← imagini brute generate original (cartoon faces)  
- `data/processed/generated_shapes_jpg/`  ← imagini 64×64 pentru RN în LabVIEW  
- `data/processed/generated_shapes_npy/`  ← imagini 64×64 normalizate pentru antrenare RN în Python  

**Dovezi:**  
- Grafic comparativ: `docs/generated_vs_public_statistics.png`  
  *Compară distribuția pixelilor între imaginile publice (fețe sintetice) și imaginile generate original (cartoon faces).*

- Setup generare (opțional): `docs/acquisition_setup.jpg`  
  *Captură de ecran a generatorului Python (nu există setup fizic deoarece datele sunt generate procedural).*

- Tabel statistici: `docs/data_statistics.csv`  
  *Statistici descriptive (medie, deviație standard, min/max, quartile) pentru dataset-ul generat și preprocesat.*

  ### 3. Diagrama State Machine a Întregului Sistem

  ### Justificarea State Machine-ului ales

Am ales o arhitectură de tip *clasificare batch + inferență sincronă* deoarece proiectul meu presupune procesarea unor imagini (fețe sintetice și forme geometrice cartoonizate), aplicarea unui flux de preprocesare și utilizarea unei rețele neuronale pentru a clasifica forma feței în trei clase: **RoundFace**, **OvalFace**, **LongFace**. Sistemul include atât module Python (generator + preprocesare + arhitectură RN), cât și componente LabVIEW (încărcare date, inferență, vizualizare rezultate).  

Scopul state machine-ului este să asigure un flux clar de la date brute → preprocesare → inferență → afișare → logare, gestionând și stările de eroare tipice ce pot apărea între Python și LabVIEW.


### Stările principale

1. **IDLE**  
   Sistemul este pornit, dar nu procesează nimic. Așteaptă comenzi de la utilizator sau un trigger de procesare.

2. **GENERATE_DATA (Python)**  
   Se rulează scriptul `generate_cartoon_faces.py` care creează datele originale pentru cele 3 clase. Aceste date reprezintă contribuția ta originală (>40% din dataset). Fiecare exemplu este o imagine generată procedural prin OpenCV.

3. **PREPROCESS (Python)**  
   Scriptul `preprocess_generated_shapes.py` preia imaginile brute, le resizează la 64×64, le normalizează, și le salvează fie ca JPG (pentru LabVIEW), fie ca NPY (pentru Python). În această etapă se verifică integritatea fișierelor și structura datasetului final.

4. **LOAD_MODEL (LabVIEW + Python backend)**  
   Modelul RN este definit și compilat în Python (`src/neural_network/model.py`). LabVIEW încarcă modelul, fie direct prin Python Node, fie printr-o interfață JSON/REST. Nu este necesar ca modelul să fie antrenat — doar încărcat fără erori.

5. **INFERENCE (LabVIEW)**  
   Utilizatorul selectează o imagine (JPG 64×64). Imaginea este dată modelului RN, care returnează o etichetă:  
   - 0 → RoundFace  
   - 1 → OvalFace  
   - 2 → LongFace  

6. **DISPLAY_RESULT (UI)**  
   Rezultatul inferenței este afișat în UI-ul creat în LabVIEW sau în interfață web (Streamlit / Gradio). Momentan este suficient un mesaj:  
   “Clasă detectată: OvalFace”.

7. **LOG_RESULT**  
   Se salvează într-un fișier CSV minimal informații precum: timestamp, numele imaginii, rezultatul inferenței. Acest log este obligatoriu în Etapa 4.

8. **ERROR**  
   Stare dedicată gestionării erorilor posibile: fișiere lipsă, corrupt, model încărcat greșit, lipsă conexiune cu Python, etc. Din această stare se revine în IDLE.

### Tranziții critice

- **IDLE → GENERATE_DATA**  
  → când utilizatorul apasă „Generate Dataset” sau prima rulare a sistemului.

- **GENERATE_DATA → PREPROCESS**  
  → după generarea cu succes a imaginilor, scriptul trece automat la preprocesare.

- **PREPROCESS → LOAD_MODEL**  
  → când toate imaginile sunt salvate în formatele necesare (JPG și NPY).

- **LOAD_MODEL → INFERENCE**  
  → când modelul RN este încărcat fără erori și poate primi input.

- **INFERENCE → DISPLAY_RESULT**  
  → imediat ce modelul returnează eticheta.

- **DISPLAY_RESULT → LOG_RESULT**  
  → după afișare, rezultatul este salvat pentru audit.

- **ANY_STATE → ERROR**  
  → dacă apar probleme la citire fișier, lipsă model, incompatibilitate shape 64×64, etc.

### De ce este starea ERROR obligatorie?

În proiectul meu, interacțiunea între Python și LabVIEW poate genera multiple probleme:  
- modelul nu se încarcă (weights random / lipsă fișier)  
- imagine coruptă  
- dimensiune greșită  
- path invalid în dataset  
- lipsă conexiune între LabVIEW și Python Node  

Starea ERROR este responsabilă pentru captarea acestor situații și revenirea în siguranță în IDLE fără a opri sistemul.

### Bucla de feedback

În arhitectura mea, feedback-ul nu actualizează procesul industrial, dar permite repetarea inferenței fără restartul aplicației — rezultatul salvat în log poate fi folosit ulterior pentru analiza performanței sau pentru recalibrarea datasetului.

## 4. Scheletul Complet al celor 3 Module Cerute în Etapa 4

Conform cerințelor cursului (slide 7), în această etapă am implementat **un schelet funcțional al întregului SIA**, compus din:

- **Modul 1:** Data Logging / Data Acquisition  
- **Modul 2:** Neural Network Module  
- **Modul 3:** Web Service / User Interface  

Toate modulele rulează fără erori și demonstrează fluxul complet de la date → preprocesare → inferență → afișarea rezultatului.


### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Acesta este modulul care se ocupă de **generarea contribuției originale a datasetului** (≥ 40%).  
În proiectul meu, datasetul original este generat printr-un **script Python propriu**, care creează imagini cartoonizate pe baza unor forme geometrice.

### ✔️ Funcționalități implementate
- Generare automată a 3 clase originale:
  - `RoundFace` (40 imagini)
  - `OvalFace` (40 imagini)
  - `LongFace` (40 imagini)
- Salvare structurată în `data/raw/generated_shapes/`
- Preprocesare automată:
  - resize la 64×64 px
  - normalizare 0–1
  - salvare JPG (pentru LabVIEW)
  - salvare NPY (pentru Python)
- Log minimal generat automat (nume imagine, clasă, timestamp)

## 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

În Etapa 4, toate cele 3 module trebuie să pornească fără erori și să existe sub forma unui schelet funcțional.  
Nu este necesară implementarea finală sau performanță ridicată — accentul este pe arhitectură, flux și modularitate.

| Modul | Python (exemple tehnologii) | LabVIEW | Cerință minimă funcțională (Etapa 4) |
|-------|------------------------------|---------|--------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | (opțional) | **MUST:** Produce date originale (minim 40%). Cod rulează fără erori și generează automat imaginile pentru cele 3 clase. |
| **2. Neural Network Module** | `src/neural_network/model.py` | (opțional pentru această etapă) | **MUST:** Modelul RN este definit și compilat corect. Nu este necesar să fie antrenat. |
| **3. Web Service / UI** | (opțional în Etapa 4) | Structură UI pregătită (fără implementare completă) | **MUST:** Există scheletul UI/fluxului (folder + descriere + plan). NU este necesar ca UI să fie complet funcțional. |

---

### Modul 1: Data Logging / Acquisition

Acest modul este **complet implementat** și reprezintă contribuția ta originală la dataset (≥ 40%).

**Funcționalități implementate:**
- ✔️ Generator propriu de imagini cartoonizate (3 clase × 40 imagini)
- ✔️ Variabilitate random pentru:
  - forma feței (round/oval/long)
  - ochi, nas, gură, sprâncene
  - culoarea/forma părului
- ✔️ Preprocesare Python:
  - resize 64×64
  - normalizare pixel 0–1
  - salvare NPY (pentru RN în Python)
  - salvare JPG (pentru LabVIEW sau UI)

**Locația codului:**  
`src/data_acquisition/`

**Locația datelor:**  
`data/raw/generated_shapes/`  
`data/processed/generated_shapes_jpg/`  
`data/processed/generated_shapes_npy/`

**Status modul:** ✔️ Funcțional integral

### Modul 2: Neural Network Module

Acest modul este **scheletul rețelei neuronale**, conform cerințelor Etapa 4.

**Funcționalități implementate:**
- ✔️ Model CNN definit în `model.py`
- ✔️ Funcție de compilare (Adam + categorical_crossentropy)
- ✔️ Arhitectură compatibilă cu imagini 64×64×3
- ✔️ Output pentru 3 clase

**Ce NU este necesar în Etapa 4:**
- antrenare reală
- metrici bune
- salvare model antrenat

**Locația codului:**  
`src/neural_network/model.py`

**Status modul:** ✔️ Complet pentru Etapa 4

### Modul 3: Web Service / UI

Pentru Etapa 4, **UI-ul nu trebuie implementat complet**.

Este suficient să existe:
- structura modulului (`src/app/`)
- o descriere clară a fluxului planificat
- un prototip minimal / placeholder

**Ce există în proiect:**
- ✔️ Folder dedicat UI: `src/app/`
- ✔️ Plan clar pentru UI (descris mai jos)
- ❗ UI-ul NU este încă implementat.

**Flux UI planificat (schelet):**
1. Utilizatorul selectează o imagine 64×64 (JPG).
2. UI apelează funcția Python care încărcă modelul neantrenat.
3. Modelul returnează clasa:  
   **RoundFace / OvalFace / LongFace**
4. UI afișează rezultatul.

**Ce voi implementa în Etapa 5/6:**
- interfață LabVIEW cu buton „Load Image”
- Python Node → apel către modelul CNN
- afișare rezultat

**Status modul:** ⚠️ Schelet pregătit (conform cerințelor Etapa 4)


## ✔️ Concluzie Secțiunea 4

Chiar dacă UI-ul nu este încă implementat complet, toate cele 3 module:
- există în structură
- sunt documentate
- sunt funcționale în limitele cerute pentru Etapa 4
- proiectul pornește fără erori și are pipeline complet în Python

proiect-rn-Rotaru-Alexandra/
├── data/
│   ├── raw/
│   │   └── generated_shapes_raw/        
│   ├── processed/
│   │   ├── generated_shapes_jpg/        
│   │   └── generated_shapes_npy/        
│   ├── generated/                      
│   ├── train/                          
│   ├── validation/                     
│   └── test/                            
│
├── src/
│   ├── data_acquisition/
│   │   ├── generate_cartoon_faces.py    
│   │   └── README.md                   
│   ├── preprocessing/
│   │   └── preprocess_generated_shapes.py  
│   ├── neural_network/
│   │   └── model.py                    
│   └── app/
│       └── ui_stub.py                   
│
├── docs/
│   ├── state_machine.png                
│   ├── screenshots/
│   │   └── ui_demo.png                  
│   └── dataset_structure.png            
│
├── models/
│   └── untrained_model.h5               
│
├── config/
│   └── config.yaml                      
│
├── README.md
├── README_Etapa4_Arhitectura_SIA.md     
└── requirements.txt

Diferențe față de Etapa 3
În Etapa 4 am adăugat:

data/generated/ – imaginile tale originale generate procedural

src/data_acquisition/ – modulul care generează datele Round/Oval/Long

src/neural_network/ – arhitectura RN neantrenată (model skeleton)

src/app/ – UI minimal (Streamlit/Gradio) pentru a testa pipeline-ul

models/ – modelul neantrenat (salvat după .compile())

docs/state_machine.png – diagrama obligatorie

docs/screenshots/ – captură UI demonstrativă

Documentație și Structură

 Tabelul Nevoie → Soluție → Modul completat
→ (clasificare forme față Round/Oval/Long)

 Declarația contribuției 40% date originale completată
→ ai 120 imagini generate 100% original

 Cod generare/achiziție funcțional
→ generate_cartoon_faces.py rulează fără erori

 Diagrama State Machine adăugată în /docs/state_machine.png

 Legendă State Machine scrisă în README_Etapa4

 Structură repository conform cerințelor Etapei 4

⚠ Nu ai nevoie de grafice comparative sau statistici pentru Etapa 4 (opțional).

✔ Modul 1: Data Logging / Acquisition

 Scriptul generate_cartoon_faces.py rulează fără erori

 Generează minim 120 imagini originale (40%+ din dataset)

 Pozele sunt salvate în data/generated/

 README în src/data_acquisition/ completat, cu:

 metoda de generare (OpenCV procedural)

 parametri random (poziție ochi, nas, gura, contur etc.)

 motivație pentru relevanță în proiectul tău

✔ Modul 2: Neural Network

 Arhitectura RN (CNN mică) definită în model.py

 Modelul este compilat fără erori

 Poate fi salvat și reîncărcat (model.save())

 README în src/neural_network/ care:

explică arhitectura

justifică dimensiunea 64×64

spune clar că modelul nu este antrenat încă (Etapa 5/6)

✔ Modul 3: Web Service / UI (schelet)

 Ai creat un UI simplu în src/app/ui_stub.py

 UI pornește fără erori (Streamlit sau Gradio)

 Acceptă un fișier JPG 64×64

 Afișează rezultatul inferenței (chiar dacă modelul nu e antrenat)

 Screenshot adăugat în docs/screenshots/ui_demo.png

 README în src/app/ cu:

instrucțiuni de rulare (streamlit run ui


