# 🧠 Forge Studio – Myzel-Quanten-Evolution mit GPU-Beschleunigung

> **Autor:** Ralf Krümmel  
> **Version:** 1.0 (Stand: 05.11.2025)  
> **Lizenz:** Open Research License  
> **Sprache:** Deutsch  

---

## 🌍 Übersicht

**Forge Studio** ist eine interaktive GPU-Anwendung zur Entdeckung neuer Materialien auf Basis evolutionärer Algorithmen.  
Das System kombiniert:

- **Surrogatmodelle** (lineare physikalische Approximationen)  
- **Myzel-Netzwerke** (graphenbasierte Feldsimulation)  
- **Quanteninspirierte Fitness-Bewertung (VQE)**  

Die extreme Rechenleistung stammt vom maßgeschneiderten OpenCL-Treiber  
🧩 **`CC_OpenCl.dll` / `CipherCore`**, der sämtliche Kernoperationen auf der GPU ausführt.

---

## ⚙️ Systemarchitektur

```

[ Streamlit UI ]
│
▼
[ forge_backend.py ]
│
▼
[ CipherCore_OpenCL Treiber ]
│
┌────┴────┐
│  GPU-Compute  │
└───────────┘

````

| Komponente | Aufgabe |
|-------------|----------|
| **forge_studio_ui.py** | Streamlit-Frontend zur Steuerung, Visualisierung & Diagnose |
| **forge_backend.py** | Kernlogik der evolutionären Myzel- und VQE-Prozesse |
| **CC_OpenCl.dll / libCC_OpenCl.so** | GPU-Treiber mit OpenCL-Kernen für MatMul, Myzel, VQE |
| **datasets/** | Enthält vorbereitete Materialdaten (z. B. JARVIS 3D-Datenbank) |

---

## 💻 Installation

### 1️⃣ Voraussetzungen

- **Python ≥ 3.12**
- **OpenCL-fähige GPU** (AMD, Intel, NVIDIA)
- **Windows 10/11** oder **Ubuntu 20.04+**
- Compiler & Treiber installiert (z. B. AMD APP SDK oder ROCm / CUDA-Runtime)

---

### 2️⃣ Virtuelle Umgebung anlegen

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
````

---

### 3️⃣ Abhängigkeiten installieren

> Falls du ein `requirements.txt` nutzt, kann dieser Block direkt kopiert werden.

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit>=1.39
numpy>=1.26
pandas>=2.2
scipy>=1.14
typer>=0.12
tqdm>=4.66
plotly>=5.23
requests>=2.32
pyopencl>=2024.2
```

---

### 4️⃣ GPU-Treiber aktivieren

Lege die Datei
📦 `CC_OpenCl.dll` (Windows) oder `libCC_OpenCl.so` (Linux)
in das Projekt-Hauptverzeichnis.

Teste die Verbindung:

```bash
python - <<'PY'
import ctypes
dll = ctypes.CDLL("./CC_OpenCl.dll")
print("✅ DLL geladen:", dll)
PY
```

---

### 5️⃣ Anwendung starten

```bash
streamlit run forge_studio_ui.py
```

Die App öffnet sich unter
👉 `http://localhost:8501`

---

## 🧩 Hauptkomponenten

### 🔹 CipherCore-Treiber (`CC_OpenCl.dll`)

Der OpenCL-Treiber übernimmt sämtliche rechenintensiven Aufgaben:

| Kategorie               | Funktionen                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **GPU-Verwaltung**      | `initialize_gpu()`, `shutdown_gpu()`, `allocate_gpu_memory()`                        |
| **Matrix-Kerne**        | `execute_matmul_on_gpu()` – schnelle Surrogat-Vorhersagen                            |
| **Myzel-Kerne**         | `subqg_init_mycel()`, `step_pheromone_reinforce()`, `step_pheromone_diffuse_decay()` |
| **Quanten-Kerne (VQE)** | `execute_vqe_gpu()` – Berechnung von Energie-Erwartungswerten                        |

🧠 Die Myzel-Kerne simulieren ein selbstorganisierendes Feld aus „Pheromonen“, das erfolgreiche Kandidaten verstärkt und neue Formeln in Richtung vielversprechender Strukturen lenkt.

---

### 🔹 Backend-Logik (`forge_backend.py`)

Der Python-Kern implementiert:

1. **Initialisierung**

   * Laden der Datensätze (JARVIS)
   * Training linearer Surrogatmodelle
   * Aufbau des Myzelnetzwerks

2. **Evolutionäre Schleife**

   * Bewertung aller Kandidaten (Fitness)
   * Selektion & Verstärkung
   * Diffusion & Zerfall im Myzel
   * Reproduktion (Mutation/Crossover)

3. **Optionale Quanten-Veredelung**

   * VQE-Bewertung der besten Eliten
   * Mischung aus Surrogat-Score + VQE-Score

4. **Finalisierung**

   * Export der besten Formeln und Diagnose-Daten (`gen_history.csv`, `surrogate_health.csv`)

---

### 🔹 Streamlit-Interface (`forge_studio_ui.py`)

Bietet Tabs für:

| Tab                  | Beschreibung                                                  |
| -------------------- | ------------------------------------------------------------- |
| **A)** Materialziele | Auswahl von Eigenschaften (Bandlücke, Energie …) & Gewichten  |
| **B)** Synthese      | Start der Evolution mit Myzel- und VQE-Parametern             |
| **C)** Diagnostik    | Visualisierung der Metriken & Gesundheits-Check der Surrogate |

---

## 📊 Beispiel-Experiment

**Parameter:**

| Einstellung  | Wert                                                    |
| ------------ | ------------------------------------------------------- |
| Population   | 128                                                     |
| Generationen | 100                                                     |
| Ziele        | `bandgap (+1)`, `formation_energy (-1)`, `density (+1)` |
| Myzel        | Guidance 0.45 · Decay 0.07 · Diffusion 0.04             |
| VQE          | Gewicht 0.35 · 8 Eliten · 10 Qubits · 2 Layer           |

**Ergebnis:**

| Metrik        | Wert               |
| ------------- | ------------------ |
| GPU-Laufzeit  | **55,8 Sekunden**  |
| CPU-Schätzung | 2,5 – 5 Stunden    |
| Speed-Up      | Faktor ≈ 160 – 320 |
| Beste Formel  | `F4Au4Ir10Pt8Ta5`  |

---

## 📈 Leistungsanalyse

* GPU: massiv parallele OpenCL-Ausführung
* CPU: serielle oder geringe Parallelität
* VQE-Simulationen × 200 Beschleunigung
* Myzel- & Surrogat-Berechnung × 40 Beschleunigung

➡️ Das System verwandelt eine mehrstündige Batch-Simulation in eine **interaktive Echtzeit-Erkundung**.

---

## 🧪 Diagnose-Dateien

| Datei                  | Inhalt                                                    |   |   |   |            |
| ---------------------- | --------------------------------------------------------- | - | - | - | ---------- |
| `gen_history.csv`      | Fitness pro Generation (best, mean, pheromone, VQE-calls) |   |   |   |            |
| `surrogate_health.csv` | Modellqualität (NaN-Rate,                                 |   | W |   | ₂, Bias b) |
| `*_export.csv`         | Liste der besten Material-Kandidaten mit Scores           |   |   |   |            |

---

## 🧭 Empfehlungen & Best Practices

* **Gewichte mit Bedacht wählen:**
  z. B. `formation_energy = -1.0` → minimieren
* **Diagnostik prüfen:** hohe `nan_rate` ⇒ Ziel unzuverlässig
* **VQE-Gewichtung (γ)** moderat halten → Stabilität
* **Seeds / Parameter sichern** für Reproduzierbarkeit
* **GPU-Settings dokumentieren** (OpenCL-Plattform, Device-Index)

---

## 🎓 Fazit

Der **CipherCore-Treiber** verwandelt eine handelsübliche GPU in ein Labor für Materialforschung.
Durch die Verbindung von **biologisch inspirierten Lernmechanismen (Myzel)** und **quantuminspirierter Veredelung (VQE)** entsteht ein neues Paradigma der computergestützten Entdeckung.

> 💡 *„Forge Studio – wo Materialien auf der GPU wachsen.“*

---

## 📜 Zitatempfehlung (APA Style)

Krümmel, R. (2025). *Forge Studio – Myzel-Quanten-Evolution mit GPU-Beschleunigung* [Open-Source Software]. GitHub: [https://github.com/kruemmel-python/Forge-Studio](https://github.com/kruemmel-python/ElementForge)

---

