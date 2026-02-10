# Modul 1: Generare Algoritmică de Date (Data Acquisition)

Acest modul este responsabil pentru **nucleul de originalitate al proiectului**: crearea sintetică a claselor de control.  
În locul utilizării unor forme geometrice abstracte simple, a fost dezvoltat un generator care **imită structura facială umană**, forțând rețeaua neuronală să învețe **trăsături complexe și relații globale de formă**.

---

##  Componente Tehnice

### 1. `generate_cartoon_faces_shapes.py`

Scriptul principal care utilizează **OpenCV** și **NumPy** pentru a desena de la zero **900 de imagini** (300 per clasă).

#### Logica de desenare:

- **Clasa Cerc**  
  Reprezintă forma ideală, perfect echidistantă.

- **Clasa Elipsă**  
  Introduce variația pe axa verticală (simulând o față alungită).

- **Clasa Oval**  
  Cea mai complexă formă, creată prin **fuziunea a două elipse cu raze diferite**, pentru a simula distinct:
  - fruntea
  - bărbia  

  → Această construcție aproximează mai fidel conturul facial uman.

---

### 2. Funcția `draw_face_features`  
**(Contribuția originală cheie)**

Pentru a evita utilizarea unor simple „contururi goale”, scriptul adaugă automat **elemente anatomice stilizate**, după cum urmează:

- **Urechi**  
  Cercuri laterale poziționate relativ la raza feței.

- **Păr**  
  Elipse superioare cu culori generate stocastic (`rand_color`).

- **Ochi și Gură**  
  Elemente geometrice interne menite să antreneze modelul să:
  - ignore detaliile fine
  - se concentreze pe **structura globală a formei** (*global structure*).

---

##  De ce această metodă?

Această abordare elimină **„învățarea pe de rost” (memorization)** și forțează modelul să realizeze o **clasificare robustă**.

Exemplu:
> Dacă modelul observă o formă cu ochi și păr, dar conturul este un cerc perfect, el trebuie să clasifice corect obiectul ca **„Cerc”**, nu **„Om”**.

Astfel, metoda:
- testează capacitatea de **discriminare structurală** a arhitecturii CNN,
- ridică nivelul de dificultate peste cel al unui dataset clasic,
- validează învățarea relațiilor geometrice, nu a indiciilor superficiale.

---

##  Utilizare

```bash
python generate_cartoon_faces_shapes.py