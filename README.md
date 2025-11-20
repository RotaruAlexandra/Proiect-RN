# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Rotaru Elena-Alexandra 
**Data:** 20.11.2025



## Introducere

În această etapă a proiectului au fost analizate și pregătite datele necesare instruirii rețelei neuronale, cu respectarea tuturor normelor privind protecția datelor personale. Setul de date utilizat nu conține imagini cu persoane reale, ci este compus exclusiv din două categorii de date non-GDPR:

Imagini sintetice generate în LabVIEW, reprezentând forme geometrice (cercuri, ovale), folosite pentru a testa funcționalitatea rețelei neuronale și întregul flux de preprocesare.

Imagini cu fețe generate artificial, obținute din surse publice precum thispersondoesnotexist.com, generated.photos sau dataset-uri Kaggle cu fețe sintetice. Aceste imagini sunt create prin algoritmi generativi și nu aparțin unor persoane reale, eliminând orice risc asociat prelucrării de date cu caracter personal.

Prin combinarea acestor două tipuri de date, etapa urmărește construirea unui set de date hibrid adecvat prototipării, analizării și preprocesării în vederea instruirii rețelelor neuronale din proiect. Activitățile realizate includ analiza exploratorie (EDA), verificarea calității datelor, normalizarea și împărțirea în seturi de tip train/validation/test, precum și organizarea completă a repository-ului GitHub conform cerințelor


##  1. Structura Repository-ului Github (versiunea Etapei 3)

recunoastere-faciala-rn/
├── README.md                     # descriere generală + Etapa 3
├── docs/
│   ├── datasets/                 # descriere surse date, GDPR-safe, diagrame
│   
├── data/
│   ├── raw/
│   │   ├── labview_shapes/       # imagini sintetice generate în LabVIEW
│   │   └── synthetic_faces/      # fețe generate artificial (non-GDPR)
│   ├── processed/                # imagini preprocesate (resize, grayscale)
│   ├── train/                    # set de antrenare
│   ├── validation/               # set de validare
│   └── test/                     # set de testare
├── src/
│   ├── preprocessing/            # scripturi Python/LabVIEW pentru preprocesare
│   ├── data_generation/          # VI-uri LabVIEW pentru generarea formelor
│   ├── detection_python/         # YOLOv11 + face_recognition (etapele următoare)
│   └── neural_network/           # RN LabVIEW (în etapa următoare)
├── config/
│   └── preprocessing_config.json # parametri: resize, grayscale, normalizare
└── requirements.txt              # dependențe Python (YOLO, OpenCV etc.)


##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

Origine:
Setul de date utilizat în proiect provine exclusiv din surse software, fără colectarea unor date reale despre persoane. Sunt folosite două tipuri de date complet non-GDPR:

Imagini sintetice generate în LabVIEW, reprezentând forme geometrice (cercuri, ovale), folosite pentru testarea fluxului de preprocesare și pentru antrenarea rețelei neuronale.

Imagini cu fețe generate artificial, obținute din surse publice precum thispersondoesnotexist.com, generated.photos și dataset-uri Kaggle cu fețe create prin algoritmi generativi (GAN). Aceste imagini nu reprezintă persoane reale și sunt permise în context academic.

Modul de achiziție:
Generare programatică (LabVIEW – figuri geometrice)
Fișier extern / dataset online (imagini cu fețe sintetice)

Perioada / condițiile colectării:Datele au fost colectate și organizate în perioada Noiembrie 2025 – Decembrie 2025.
Imaginile din LabVIEW sunt generate în condiții complet controlate (fundal simplu, formă clară, rezoluție uniformizată).
Imaginile cu fețe sintetice sunt descărcate din surse publice gratuite, create artificial, utilizate exclusiv pentru scopuri educaționale.


### 2.2 Caracteristicile dataset-ului
Număr total de observații: aproximativ 40–80 de imagini
– ~20 imagini sintetice generate în LabVIEW (cercuri, ovale)
– ~20–60 imagini cu fețe generate artificial (non-GDPR)

Număr de caracteristici (features): în funcție de preprocesare:
– ~4.096 elemente pentru imagini resize 64×64 (vector flatten)
– + 1 caracteristică pentru clasa/eticheta imaginii ("cerc", "oval", "fata_1", "fata_2" etc)
Tipuri de date:
Imagini (date vizuale)
Date numerice (valori de intensitate pixel grayscale după preprocesare)
Categoriale (doar eticheta de clasă)
Temporale (nu se utilizează)

Format fișiere:
PNG / JPG pentru imaginile originale (raw)
CSV pentru vectorii de trăsături extrași în etapa de preprocesare
DIR structurat (train/validation/test) pentru folderele finale

| **Caracteristică** | **Tip**    | **Unitate**                     | **Descriere**                                                                         | **Domeniu valori**                                        |
| ------------------ | ---------- | ------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| pixel_value_i      | numeric    | 0–255 (raw)<br>0–1 (normalizat) | Valoarea de intensitate a pixelului în imaginea grayscale, după vectorizarea imaginii | Pixel grayscale: 0 (negru) – 255 (alb)<br>Normalizat: 0–1 |
| width              | numeric    | px                              | Lățimea imaginii după preprocesare (resize)                                           | {64, 128, …} în funcție de rezoluția aleasă               |
| height             | numeric    | px                              | Înălțimea imaginii după preprocesare                                                  | {64, 128, …} în funcție de rezoluția aleasă               |
| norm_pixel_value_i | numeric    | –                               | Pixel normalizat între 0 și 1, utilizat ca input pentru RN                            | 0–1                                                       |
| class_label        | categorial | –                               | Eticheta imaginii: formă geometrică sau identitate sintetică                          | {cerc, oval, fata_1, fata_2, …}                           |
| file_source        | categorial | –                               | Sursa imaginii: LabVIEW sau generator artificial de fețe                              | {labview_shape, synthetic_face}                           |
pixel_value_i

Fiecare pixel devine o caracteristică numerică.
Dacă imaginea are 4096 pixeli → ai 4096 astfel de valori.

• norm_pixel_value_i

Este aceeași valoare, dar scalată în intervalul 0–1 (necesară pentru RN).

• class_label

Distinge între:

forme geometrice generate în LabVIEW (ex: cerc, oval)

fețe generate AI (fata_1, fata_2 etc.)

• file_source

Arată sursa datelor → foarte important pentru documentarea dataset-ului.

Fișier recomandat: data/README.md



##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

În cadrul acestei etape, analiza exploratorie a fost realizată pe valorile de intensitate ale pixelilor din imaginile utilizate (forme geometrice generate în LabVIEW și fețe sintetice).

Medie, mediană, deviație standard
– aplicate asupra intensităților pixelilor pentru a identifica nivelul general de luminanță al imaginilor și variația acestora între clase (forme vs. fețe).

Min–max și quartile (Q1, Q2, Q3)
– folosite pentru a evalua distribuția globală a pixelilor grayscale și pentru a verifica dacă imaginile sunt supraexpuse, subexpuse sau corect preprocesate.

Distribuții pe caracteristici (histograme grayscale)
– au fost generate histograme pentru imaginile sintetice și pentru fețele artificiale pentru a observa diferențele de contrast, iluminare și uniformitate a datelor.

Identificarea outlierilor (IQR / percentile)
– utilizată pentru a detecta imagini cu anomalii (ex: complet întunecate, complet albe, cu zgomot vizual), care ar putea afecta negativ antrenarea rețelei neuronale.


### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** ...
* **Detectarea valorilor inconsistente sau eronate**  ....
* **Identificarea caracteristicilor redundante sau puternic corelate** ...

### 3.3 Probleme identificate

În urma analizei preliminare asupra setului de imagini brute (LabVIEW + imagini sintetice), au fost identificate următoarele probleme:

Rezoluții neuniforme
– imaginile generate în LabVIEW au dimensiuni constante, însă fețele sintetice descărcate au rezoluții diferite → este necesară redimensionarea la o rezoluție standard (ex. 64×64 px).

Iluminare și contrast variabil
– imaginile cu fețe sintetice prezintă variații mari ale luminozității → este necesară normalizare.

Clase dezechilibrate
– la început vor exista mai multe fețe decât forme geometrice → se va echilibra setul prin generarea suplimentară de cercuri, ovale și elipse.

Fundaluri diferite
– imaginile LabVIEW au fundal alb uniform, în timp ce fețele sintetice au fundaluri complexe → preprocesare obligatorie (grayscale + normalizare).


##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Eliminare duplicatelor
– verificare hash și dimensiuni pentru a evita imagini duplicate în seturi diferite.

Tratarea valorilor lipsă / fișiere corupte
– au fost verificate imaginile brute; nu există fișiere corupte, dar procesul include o verificare automată.

Tratarea outlierilor vizuali
– eliminare sau corectare a imaginilor cu expunere extremă detectate prin analiza histogramelor.


### 4.2 Transformarea caracteristicilor
Normalizare Min–Max (0–1)
– pixelii sunt împărțiți la 255 pentru compatibilitate cu LabVIEW RN Toolkit.

Redimensionare (resize)
– toate imaginile sunt scalate la 64×64 px.

Conversie grayscale
– elimină variabile inutilizabile și reduce complexitatea.

Vectorizare (flatten)
– imagine 64×64 → vector de 4096 valori (necesar pentru RN în LabVIEW).

Encoding etichete
– ex.: cerc = 0, oval = 1, elipsă = 2, față_1 = 3 etc.

Echilibrare clase
– generare suplimentară de forme geometrice pentru a obține un dataset aproximativ echilibrat.

### 4.3 Structurarea seturilor de date

Împărțire utilizată:

70% – train

15% – validation

15% – test

Principii respectate:

Stratificare pe clase (cerc, oval, elipsă, fețe)
Fără "data leakage" (o imagine apare doar într-un singur set)
Parametrii de normalizare calculați doar pe train, apoi aplicați identic pe val/test

### 4.4 Salvarea rezultatelor preprocesării

Imaginile preprocesate sunt salvate în: data/processed/
Seturile finale sunt salvate în:

data/train/
data/validation/
data/test/

Parametrii utilizați la preprocesare (ex. rezoluție, scalare, grayscale) sunt salvați în:config/preprocessing_config.json

##  5. Fișiere Generate în Această Etapă
data/raw/ – imagini brute (LabVIEW + fețe sintetice)

data/processed/ – imagini preprocesate (resize, grayscale, normalizare)

data/train/, data/validation/, data/test/ – seturile finale pentru RN

src/preprocessing/ – scripturi Python pentru preprocesarea imaginilor

src/data_generation/ – VI-uri pentru generarea formelor geometrice în LabVIEW

data/README.md – descrierea datasetului utilizat

config/preprocessing_config.json – parametri folosiți la preprocesare


##  6. Stare Etapă 

 Structură repository configurată
 Dataset analizat (EDA realizată)
 Date preprocesate (resize, grayscale, normalizare)
 Seturi train/validation/test generate
 Documentație actualizată în README.md și data/README.md
