# Documentație Set de Date (Dataset Specifications)

Acest folder conține infrastructura de date utilizată pentru proiect.  
Setul de date este un **hibrid** creat pentru a echilibra nevoia de date reale cu necesitatea unor grupuri de control geometrice precise.

---

##  1. Structura și Distribuția Datelor

Dataset-ul este **perfect echilibrat** pentru a preveni bias-ul modelului către o anumită clasă.

| Clasă   | Nr. Imagini | Sursă                                   | Tip Date                  |
|--------|------------|------------------------------------------|---------------------------|
| Om     | 300        | Kaggle (Human Faces) + Online Generator | Real / Sintetic (Photo)  |
| Cerc   | 300        | Script propriu (OpenCV)                  | Sintetic (Geometric)     |
| Elipsă | 300        | Script propriu (OpenCV)                  | Sintetic (Geometric)     |
| Oval   | 300        | Script propriu (OpenCV)                  | Sintetic (Geometric)     |
| **TOTAL** | **1200** | —                                        | **75% Contribuție Originală** |

**Format date:**  
- Grayscale  
- Rezoluție: **64×64**

---

##  2. Pipeline de Preprocesare (Data Engineering)

Fiecare imagine a trecut printr-un flux riguros de transformare înainte de a ajunge în rețeaua neuronală:

1. **Segmentare Semantică (MediaPipe)**  
   Pentru clasa *„Om”*, a fost utilizat modelul **Selfie Segmentation** pentru eliminarea fundalului complex, înlocuindu-l cu alb pur.  
   → Scop: forțarea modelului să învețe exclusiv trăsăturile faciale.

2. **Standardizare Geometrică**  
   Redimensionare la **64×64 pixeli** folosind interpolarea `cv2.INTER_AREA`, optimizată pentru reducerea dimensiunii fără pierderi semnificative ale marginilor.

3. **Normalizare Z-Score**  
   Valorile pixelilor au fost scalate în intervalul **[-1, 1]** folosind: μ = 0.5, σ = 0.5

4. **Augmentare (Albumentations)**  
   Aplicată exclusiv pe setul de antrenare:
   - rotații
   - zgomot Gaussian
   - distorsiuni de perspectivă  

   → Crește capacitatea de generalizare a modelului.

---

##  3. Proveniență și Trasabilitate

- **Date brute:**  
  - `data/raw/` – imagini originale  
  - `data/generated/` – forme generate cu OpenCV  

- **Script generare forme:**  
  `src/data_acquisition/generate_cartoon_faces_shapes.py`

- **Script preprocesare:**  
  `src/data_acquisition/preprocess_generated_shapes.py`

- **Configurație:**  
  Parametrii de normalizare sunt salvați în  
  `config/preprocessing_params.pkl`

---

##  4. Note privind Confidențialitatea (GDPR)

Imaginile din clasa **„Om”** provin exclusiv din:
- surse publice (*Kaggle*), sau
- generare sintetică (*This Person Does Not Exist*).

Nu au fost utilizate date biometrice reale ale studenților sau personalului universitar fără consimțământ explicit, respectând **normele europene de protecție a datelor (GDPR)**.

---

 *Acest dataset este conceput strict pentru cercetare și dezvoltare experimentală în cadrul proiectului Human Campus Monitoring.*