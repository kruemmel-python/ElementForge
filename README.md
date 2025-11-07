# Forge Studio: Industrial Formulation Engine

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python-Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/gebaut%20mit-Streamlit-red.svg)](https://streamlit.io)

**GPU-beschleunigte Rezepturoptimierung für die Materialwissenschaft und Industrie mittels eines evolutionären Algorithmus, geleitet durch ein Mycelial Prototype Graph (MPG) Surrogatmodell.**

Forge Studio ist ein Framework zur Entdeckung und Optimierung von industriellen Rezepturen und neuen Materialien. Anstatt auf langwierige Laborexperimente zu setzen, nutzt dieses Projekt einen datengesteuerten Ansatz, um den Suchraum potenzieller Materialkombinationen effizient zu durchsuchen. Der Workflow reicht von der abstrakten Definition von Zielen über die KI-gestützte Suche bis hin zur Generierung konkreter, umsetzbarer Laborrezepte.



---

## Inhaltsverzeichnis

1.  [Über das Projekt](#1-über-das-projekt)
2.  [Architektur und Kernkonzepte](#2-architektur-und-kernkonzepte)
    *   [Systemarchitektur](#systemarchitektur)
    *   [CipherCore OpenCL Treiber (DLL)](#ciphercore-opencl-treiber-dll)
    *   [Mycelial Prototype Graph (MPG) Modell](#mycelial-prototype-graph-mpg-modell)
    *   [Evolutionärer Algorithmus](#evolutionärer-algorithmus)
    *   [Materials Project Datenkonnektor](#materials-project-datenkonnektor)
3.  [Installation und Einrichtung](#3-installation-und-einrichtung)
    *   [Voraussetzungen](#voraussetzungen)
    *   [Installationsanleitung](#installationsanleitung)
4.  [Benutzerhandbuch](#4-benutzerhandbuch)
    *   [Starten der Anwendung](#starten-der-anwendung)
    *   [Tab B: Evolutionäre Formulierung entwerfen](#tab-b-evolutionäre-formulierung-entwerfen)
    *   [Tab C: Diagnostik der Suche](#tab-c-diagnostik-der-suche)
    *   [Tab D: Rezept-Generator](#tab-d-rezept-generator)
5.  [Fallstudie & Testergebnisse](#5-fallstudie--testergebnisse)
    *   [Die Herausforderung: Konvergenz zu suboptimalen Ergebnissen](#die-herausforderung-konvergenz-zu-suboptimalen-ergebnissen)
    *   [Der Durchbruch: "Constrained Optimization"](#der-durchbruch-constrained-optimization)
    *   [Beispielhafte Ergebnisse](#beispielhafte-ergebnisse)
6.  [Anhang](#6-anhang)
    *   [Projektdateien](#projektdateien)
    *   [Glossar](#glossar)
    *   [Lizenz](#lizenz)

---

## 1. Über das Projekt

Das Problem, das Forge Studio löst, ist die zeitaufwändige und oft empirische Suche nach neuen Materialien oder industriellen Rezepturen mit gewünschten Eigenschaften. Durch die Kombination von GPU-Beschleunigung, Surrogatmodellierung und evolutionären Algorithmen bietet es einen datengesteuerten, intelligenten Optimierungsprozess, der den gesamten Entdeckungszyklus beschleunigt.

**Hauptmerkmale:**

*   **Vollständiger Workflow:** Von der abstrakten Zieldefinition bis zum konkreten Laborrezept.
*   **Hohe Performance:** Ein maßgeschneiderter C-Treiber (`CipherCore_OpenCl.dll`) nutzt OpenCL für massive Parallelberechnungen auf der GPU.
*   **Intelligente Suche:** Ein genetischer Algorithmus, geleitet durch ein Mycelial Prototype Graph (MPG) Surrogatmodell, erkundet effizient den Lösungsraum.
*   **Robuste Datenanbindung:** Ein flexibler Konnektor bindet die Materials Project API an, inklusive Fallback-Mechanismen und umfassender Datenbereinigung.
*   **Interaktive Benutzeroberfläche:** Eine mit Streamlit erstellte Web-App ermöglicht die intuitive Steuerung aller Parameter und die Visualisierung der Ergebnisse.

## 2. Architektur und Kernkonzepte

### Systemarchitektur

Das Projekt ist in drei klare Schichten unterteilt, die eine saubere Trennung von Darstellung, Logik und Berechnung gewährleisten:

1.  **Präsentationsschicht (`forge_studio_ui.py`):** Eine interaktive Web-UI (Streamlit).
2.  **Logikschicht (`forge_backend.py`):** Das Python-Herzstück, das den evolutionären Algorithmus und das Modelltraining steuert.
3.  **Compute-Schicht (`CipherCore_OpenCl.dll`):** Die in C geschriebene High-Performance-Bibliothek für alle rechenintensiven GPU-Aufgaben.

```
[ UI (Streamlit) ] <--> [ Python Backend (Logik) ] <--> [ CipherCore DLL (C/OpenCL) ] <--> [ GPU ]
```

### CipherCore OpenCL Treiber (DLL)

Die `CipherCore_OpenCl.dll` ist das Fundament der hohen Performance. Diese in C geschriebene Bibliothek ist verantwortlich für:
*   **GPU-Initialisierung:** Erkennt und verbindet sich mit OpenCL-fähigen Grafikkarten.
*   **Kernel-Kompilierung:** Kompiliert spezialisierte C-Programme (Kernel) zur Laufzeit für die spezifische GPU-Architektur.
*   **Speicherverwaltung:** Allokiert GPU-Speicher und managt den Datentransfer zwischen CPU und GPU.
*   **C-API:** Stellt eine Schnittstelle bereit, die von Python über `ctypes` aufgerufen wird, um GPU-Berechnungen anzustoßen.

### Mycelial Prototype Graph (MPG) Modell

Das MPG-Modell ist ein Surrogatmodell, das lernt, die Ergebnisse teurer Experimente oder Simulationen zu approximieren.
*   **Prototype Graph:** Das Modell lernt eine Reihe von "Prototypen" – ideale Repräsentanten für bestimmte Materialcluster. Die Eigenschaften neuer Kandidaten werden durch Vergleich mit diesen Prototypen vorhergesagt.
*   **Mycelial Guidance:** Ein Pheromon-System leitet den evolutionären Algorithmus. Wie ein Pilzmyzel hinterlässt der Algorithmus "Spuren" in vielversprechenden Bereichen des Suchraums.

### Evolutionärer Algorithmus

Der Optimierungsprozess folgt den Prinzipien der natürlichen Selektion:
1.  **Initialisierung:** Erzeugung einer zufälligen "Population" von Kandidaten.
2.  **Bewertung (Fitness):** Das MPG-Modell sagt die Eigenschaften jedes Kandidaten voraus, und eine Funktion berechnet einen Fitness-Score basierend auf den Zielen.
3.  **Selektion & Reproduktion:** Die besten Kandidaten überleben und erzeugen durch Kreuzung (`crossover`) und Mutation neue Nachkommen.
4.  **Wiederholung:** Dieser Zyklus wird über viele Generationen wiederholt, wodurch sich die Population schrittweise in Richtung optimaler Lösungen entwickelt.

### Materials Project Datenkonnektor

Das Modul `materials_mp_connector.py` ist ein flexibler und robuster Datenkonnektor für die Materials Project API.
*   **Dual-Mode-Fähigkeit:** Kann wahlweise den offiziellen Python-Client (`mp-api`) oder einen generischen REST-Ansatz (`requests`) verwenden, inklusive automatischem Fallback.
*   **Umfassende Datenbereinigung:** Führt Spaltenumbenennung, Typkonvertierung, intelligente Handhabung fehlender Werte und Imputation durch, um ein sauberes, sofort nutzbares Pandas DataFrame zu liefern.
*   **Flexibilität:** Die `ColumnMap`-Struktur ermöglicht eine einfache Anpassung der abgerufenen Daten an die Anwendungsanforderungen.

## 3. Installation und Einrichtung

### Voraussetzungen

*   Python 3.10 oder höher
*   Ein C/C++ Compiler (z.B. GCC/Clang auf Linux, MSVC auf Windows)
*   Aktuelle Treiber für eine OpenCL 1.2+ fähige GPU (NVIDIA, AMD, Intel)
*   Git zur Versionskontrolle

### Installationsanleitung

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Python-Abhängigkeiten installieren:**
    ```bash
    pip install streamlit pandas numpy scikit-learn mp-api requests
    ```

3.  **CipherCore DLL kompilieren:**
    Navigieren Sie in das Verzeichnis mit der C-Quelldatei `opencl_driver.c`.

    *   **Unter Linux/macOS (mit GCC/Clang):**
        ```bash
        # Der -I Pfad muss möglicherweise an Ihr System angepasst werden
        gcc -shared -o CipherCore_OpenCl.so -fPIC opencl_driver.c -lOpenCL
        ```

    *   **Unter Windows (mit MSVC in einer "Developer Command Prompt"):**
        ```bash
        # 'path\to\opencl\sdk' an den Installationsort Ihres OpenCL SDKs anpassen
        cl /LD opencl_driver.c -I"path\to\opencl\sdk\include" "path\to\opencl\sdk\lib\OpenCL.lib" /FeCipherCore_OpenCl.dll
        ```
    > **Wichtig:** Stellen Sie sicher, dass die kompilierte DLL/SO-Datei im Hauptverzeichnis des Projekts liegt, damit sie von der automatischen Erkennung gefunden wird.

## 4. Benutzerhandbuch

### Starten der Anwendung
1.  Öffnen Sie ein Terminal im Projektverzeichnis.
2.  Führen Sie den Befehl aus:
    ```bash
    streamlit run forge_studio_ui.py
    ```
3.  Die Anwendung wird in Ihrem Standard-Webbrowser geöffnet.

### Tab B: Evolutionäre Formulierung entwerfen
Dies ist der Haupt-Tab, um eine Optimierung zu starten.

1.  **Anwendungsproblem auswählen:** Wählen Sie z.B. "Stabile Halbleiter".
2.  **Ziele definieren:** Wählen Sie die zu optimierenden Eigenschaften (z.B. "Bandlücke", "Bildungsenergie") und deren Gewichte (positiv = maximieren, negativ = minimieren).
3.  **Datenquelle & Parameter:**
    *   Geben Sie Ihren **Materials Project API-Key** ein.
    *   Geben Sie eine **MP Query** an (z.B. `*-O` für alle Oxide, um eine große, vielfältige Trainingsbasis zu schaffen).
    *   Stellen Sie **Population** (z.B. `512`), **Generationen** (z.B. `200`) und **Anzahl Prototypen** (z.B. `256` oder höher für mehr Modelldetail) ein.
4.  **Myzel-Parameter:** Passen Sie die `Guidance-Stärke` an, um die Balance zwischen Erkundung (niedrige Werte) und Ausnutzung (hohe Werte) zu steuern.
5.  **Synthese starten:** Klicken Sie auf den Button, um den GPU-beschleunigten Lauf zu starten. Nach Abschluss werden die Ergebnisse in einer Tabelle angezeigt und können als CSV heruntergeladen werden.

### Tab C: Diagnostik der Suche
Dieser Tab gibt Einblicke in den abgeschlossenen Lauf.
*   **Letzte Laufzeit:** Zeigt die Gesamtdauer der Berechnung.
*   **Fitness-Verlauf:** Visualisiert den Anstieg des besten Scores über die Generationen – ein Indikator für eine erfolgreiche Optimierung.
*   **Myzel-Aktivität:** Zeigt die durchschnittliche Pheromon-Stärke.

### Tab D: Rezept-Generator
Dieser Tab erscheint, sobald Ergebnisse aus Tab B vorliegen. Er übersetzt die abstrakten Ergebnisse in konkrete Materialien.
1.  **Konfigurieren:** Stellen Sie ein, wie viele Kandidaten pro Ziel gesucht und für wie viele Top-Kandidaten Rezepte erstellt werden sollen.
2.  **Mapping starten:** Klicken Sie auf den Button. Das System durchsucht das Materials Project nach realen Materialien, die Ihren optimierten Eigenschaften entsprechen.
3.  **Ergebnisse analysieren:**
    *   **Vorgeschlagene Materialien:** Zeigt eine Liste realer Materialien für jede Ihrer Top-Lösungen an.
    *   **Konsens-Ranking:** Eine aggregierte Liste der Materialien, die am häufigsten und besten zu Ihren Zielen passen.
    *   **Generierte Labor-Rezepte:** Detaillierte, schrittweise Anleitungen zur Synthese der Top-Konsens-Kandidaten, inklusive benötigter Ausgangsstoffe und deren Massen.

## 5. Fallstudie & Testergebnisse

Die Entwicklung von Forge Studio war ein iterativer Prozess, der typische Herausforderungen im Machine Learning aufzeigte.

### Die Herausforderung: Konvergenz zu suboptimalen Ergebnissen
Erste erfolgreiche Läufe zeigten ein "kollabiertes" Modell: Der Score sprang sofort auf den Maximalwert, und die Vorhersagen waren für alle Kandidaten identisch. Das Modell war nicht komplex genug und die Bewertungsfunktion zu einfach, was den Algorithmus daran hinderte, den Lösungsraum wirklich zu erkunden.

### Der Durchbruch: "Constrained Optimization"
Die finale, erfolgreiche Strategie war eine Vereinfachung der Aufgabe für die KI:
1.  **Daten filtern:** Das Modell wird ausschließlich auf **nachweislich stabilen Materialien** trainiert (`energy_above_hull <= 0.02 eV/Atom`).
2.  **Ziel fokussieren:** Das Modell wird darauf trainiert, **nur eine primäre Eigenschaft** (z.B. `band_gap`) vorherzusagen.
3.  **Score neu definieren:** Die Bewertungsfunktion kombiniert die **Modellvorhersage** (z.B. `pred_band_gap`) mit den **direkten "Genen"** des Kandidaten (z.B. `formation_energy_per_atom`), um einen echten Kompromiss zu bewerten.

### Beispielhafte Ergebnisse
Mit dieser Methode zeigt die Anwendung das gewünschte Verhalten: Der Fitness-Verlauf im Diagnostik-Tab demonstriert einen klaren Lernprozess, bei dem der Score über die Generationen hinweg ansteigt. Der Rezept-Generator liefert am Ende eine Liste konkreter, vielversprechender Materialien wie `Al2O3`, `ZrO2`, `HfO2` und `MgAl2O4` und berechnet sogar die stöchiometrisch korrekten Mengen der Ausgangsstoffe für eine Laborsynthese.

## 6. Anhang

### Projektdateien
*   `forge_studio_ui.py`: Implementiert die Streamlit-Benutzeroberfläche.
*   `forge_backend.py`: Enthält die Kernlogik des evolutionären Algorithmus und des MPG-Modells.
*   `materials_mp_connector.py`: Stellt die Verbindung zur Materials Project API her.
*   `recipe_mapper.py`: Übersetzt optimierte Eigenschaften in reale Materialien und Laborrezepte.
*   `CipherCore_OpenCl.dll` / `opencl_driver.c`: Die GPU-Compute-Bibliothek und ihr Quellcode.

### Glossar
*   **API (Application Programming Interface):** Eine Softwareschnittstelle zur Interaktion.
*   **Bandlücke (Band Gap):** Ein Maß für die Energie, die benötigt wird, um ein Elektron in einem Material leitfähig zu machen.
*   **Energie über der Hull (Energy Above Hull):** Ein Maß für die thermodynamische Stabilität eines Materials (Ziel: ≤ 0).
*   **Evolutionärer Algorithmus:** Eine von der biologischen Evolution inspirierte Optimierungsmethode.
*   **Feature:** Eine Eingabevariable für ein Machine-Learning-Modell.
*   **Kernel (OpenCL):** Ein kleines Programm, das zur Ausführung auf einer GPU geschrieben wurde.
*   **MPG (Mycelial Prototype Graph):** Der Name des in diesem Projekt verwendeten Surrogatmodells.
*   **OpenCL:** Ein offener Standard für die parallele Programmierung auf heterogenen Systemen (wie GPUs).
*   **Surrogatmodell:** Ein schnelles Machine-Learning-Modell, das als Ersatz für eine langsame Simulation dient.
*   **Paginierung:** Das Aufteilen großer Datenmengen in kleinere "Seiten" für den Abruf über eine API.

### Lizenz
Dieses Projekt ist unter der **MIT-Lizenz** lizenziert. Weitere Informationen finden Sie in der `LICENSE`-Datei.
