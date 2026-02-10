# Modul 3: Interfața Utilizator și Utilitare de Raportare

Acest modul reprezintă **nivelul de interacțiune al sistemului**, transformând modelul de inteligență artificială într-un instrument accesibil pentru **monitorizarea prezenței în campus**.  
Aplicația este construită folosind framework-ul **Streamlit**, optimizat pentru:
- afișarea rapidă a metricilor,
- prezentarea rezultatelor de inferență,
- generarea automată a rapoartelor tehnice.

---

##  Structura Modulului

| Fișier | Rol | Descriere |
|------|-----|-----------|
| `main.py` | Core Engine | Gestionează încărcarea modelului (`best_CNN_Principal.pt`), procesarea imaginilor prin *Drag & Drop* și dashboard-ul interactiv |
| `generate_results.py` | Report Generator | Script utilitar care compilează automat metricile brute în rapoarte oficiale (`.json`, `.csv`), necesare în etapele 5 și 6 |
| `style.css` | Custom UI | Fișier de stilizare avansată care implementează tema vizuală profesională (*Midnight Blue*) |

---

##  Automatizarea Rezultatelor (`generate_results.py`)

Deși este poziționat în nivelul de prezentare, acest script joacă un rol critic în **formalizarea rezultatelor experimentale**.

### Responsabilități:

- **Funcționalitate**  
  Traduce log-urile de antrenament și evaluare în formate standardizate, conforme cu template-ul proiectului.

- **Căi relative inteligente**  
  Scriptul detectează automat rădăcina proiectului și populează corect directorul `results/`, asigurând:
  - trasabilitate completă
  - consistență între model, inferență și raportare

---

##  Design Personalizat (UX / UI)

Spre deosebire de o aplicație web standard, acest modul integrează o **componentă originală de design**, optimizată pentru utilizare continuă.

### Elemente cheie:

- **Glassmorphism**  
  Cardurile de metrici sunt semi-transparente, cu efect subtil de *blue glow*, pentru evidențierea indicatorilor critici:
  - acuratețe
  - latență

- **Midnight Gradient**  
  Fundal profesional conceput pentru reducerea oboselii vizuale în centrele de monitorizare.

- **Responsive Audit Log**  
  Jurnalul de prezență se actualizează dinamic prin citirea fișierului `attendance_log.csv`.

---

##  Instrucțiuni de Utilizare

Pentru funcționarea corectă a modulului, navigați în rădăcina proiectului și executați următoarele comenzi:

### 1. Generarea rapoartelor de performanță

```bash
python src/app/generate_results.py
streamlit run src/app/main.py