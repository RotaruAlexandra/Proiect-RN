2. Descrierea Setului de Date
   
2.1 Sursa datelor

Origine (situația actuală):
În acest stadiu al proiectului, setul de date utilizat provine exclusiv din imagini de fețe sintetice generate artificial, fără a include încă imaginile create în LabVIEW.

Tipul de date utilizat în această etapă:

 Imagini cu fețe generate artificial

– surse: thispersondoesnotexist.com
– complet non-GDPR
– utilizate exclusiv în scop academic
– ~50 imagini raw

LabVIEW shapes (cercuri, ovale)
→ NU au fost încă generate, dar vor fi incluse într-o etapă ulterioară.
→ Structura de directoare și pipeline-ul Python permit deja integrarea lor.

Mod de achiziție actual:
– Descărcare online din surse publice
– Preprocesare completă în Python

Perioada colectării:
Noiembrie 2025 – Decembrie 2026

2.2 Caracteristicile dataset-ului (actual)
Număr total de imagini prelucrate: 50 imagini cu fețe sintetice (după preprocesare în Python)

Preprocesare aplicată (actual):

- Eliminare background (MediaPipe Selfie Segmentation)
- Resize la 64×64 px
- Normalizare 0–1
- Salvare în două formate:

.jpg → vizualizare

.npy → antrenare RN în Python

Dimensiunea finală a fiecărei imagini:

(64, 64, 3)

Număr de caracteristici per observație:

→ 64 × 64 × 3 = 12.288 valori (RGB)
→ sau 4096 în grayscale (dacă se va converti ulterior)

Fișiere generate:

Raw: imagini color

Preprocesate: .jpg (background removed)

Tensor RN: .npy normalizat 0–1

Caracteristicile variabilelor
Caracteristică	Tip	Descriere	Valori
pixel_value_i	numeric	intensitatea pixelului RGB	0–255
norm_pixel_value_i	numeric	intensitate normalizată	0–1
class_label	categorial	nu se folosește încă	–
file_source	categorial	„synthetic_face”	constant

În acest moment nu există etichete de clasă, deoarece am doar imagini de fețe.
Label-urile pentru formele geometrice vor fi adăugate după generarea datelor LabVIEW.

3. Analiza Exploratorie a Datelor (EDA)
   
3.1 Statistici descriptive (pe fețele sintetice)

S-au analizat:

- distribuția intensităților pixelilor (RGB → grayscale)
- media și dispersia pixelilor
- detectarea outlierilor (imagini prea întunecate / supraexpuse)
- verificarea uniformității iluminării după preprocesare

3.2 Calitatea datelor

- Fără fișiere corupte (scriptul Python verifică automat)
- Background eliminat → uniformitate îmbunătățită
- Dimensiuni raw diferite → rezolvate prin resize 64×64
- Iluminare variabilă → normalizare 0–1

3.3 Probleme identificate (actuale):

- Lipsa datelor LabVIEW (vor fi integrate ulterior)
- Lipsa etichetelor de clasă (vor fi definite după includerea formelor)
- Variabilitate ridicată a iluminării în imaginile raw
→ normalizarea a rezolvat problema

4. Preprocesarea Datelor
4.1 Curățare (actual)

- Eliminarea background-ului (MediaPipe Selfie Segmentation)
- Eliminarea duplicatelor (nume fișier + hash)
- Gestionarea fișierelor necitibile

4.2 Transformări aplicate (actual)
- Eliminare background (PAS PRINCIPAL NOU)

→ toate fețele sunt segmentate pe fundal negru/uniform

- Resize

→ 64×64 px

- Normalizare

→ pixel/255 → interval [0,1]

- Conversie RGB

→ pentru compatibilitate cu RN CNN din Python

- Salvare în două formate:

.jpg → pentru vizualizare și LabVIEW

.npy → pentru antrenare RN în Python

 Conversia grayscale + vectorizarea (flatten)

→ vor fi aplicate abia după ce ai și imaginile LabVIEW, pentru scenario RN LabVIEW

4.3 Structurarea dataset-ului (actual)

În această etapă:

imaginile sunt salvate în:

data/processed/resized_64x64_clean/
data/processed/npy_64x64_clean/


încă NU s-a realizat împărțirea în train/val/test, deoarece dataset-ul final depinde de adăugarea imaginilor LabVIEW.

4.4 Fișiere generate în această etapă

imagini preprocesate 64×64 fără background (.jpg)

tensori normalizați .npy (pregătiți pentru RN în Python)

scripturi Python în src/preprocessing/

5. Fișiere generate
 data/raw/

- imagini brute (doar fețe sintetice în acest moment)

 data/processed/

- imagini resize 64×64 + background removed

data/processed/npy_64x64_clean/

- tensori normalizați pentru RN Python

 src/preprocessing/

- scripturi pentru eliminare background + normalizare + salvare

6. Stare Etapă – ACTUALIZATĂ
- Realizate:

Preprocesare imagini sintetice (face images)

Eliminare background (MediaPipe)

Resize 64×64 px

Normalizare 0–1

Salvare .jpg + .npy

Structură repository configurată

Urmează:

Generarea forme geom. în LabVIEW

Etichetarea claselor

Vectorizare grayscale

Split train/val/test

Antrenarea RN (LabVIEW sau Python)
