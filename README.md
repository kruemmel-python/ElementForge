



# Forge Studio: Industrial Formulation Engine

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python-Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/gebaut%20mit-Streamlit-red.svg)](https://streamlit.io)

**GPU-beschleunigte Rezepturoptimierung für die Materialwissenschaft und Industrie mittels eines evolutionären Algorithmus, geleitet durch ein Mycelial Prototype Graph (MPG) Surrogatmodell.**

---

## Inhaltsverzeichnis

1.  [Über das Projekt](#über-das-projekt)
2.  [Kernkonzepte](#kernkonzepte)
    *   [CipherCore OpenCL Treiber (DLL)](#ciphercore-opencl-treiber-dll)
    *   [Mycelial Prototype Graph (MPG) Modell](#mycelial-prototype-graph-mpg-modell)
    *   [Evolutionärer Algorithmus](#evolutionärer-algorithmus)
3.  [Architektur](#architektur)
4.  [Installation und Einrichtung](#installation-und-einrichtung)
    *   [Voraussetzungen](#voraussetzungen)
    *   [Schritt-für-Schritt-Anleitung](#schritt-für-schritt-anleitung)
5.  [Anwendung und Nutzung](#anwendung-und-nutzung)
    *   [Starten der Anwendung](#starten-der-anwendung)
    *   [Konfiguration eines Optimierungslaufs](#konfiguration-eines-optimierungslaufs)
6.  [Fallstudie: Von Fehlern zur erfolgreichen Optimierung](#fallstudie-von-fehlern-zur-erfolgreichen-optimierung)
    *   [Phase 1: Technische Fehlerbehebung](#phase-1-technische-fehlerbehebung)
    *   [Phase 2: Das "kollabierte Modell"](#phase-2-das-kollabierte-modell)
    *   [Phase 3: Verfeinerung der Bewertungslogik](#phase-3-verfeinerung-der-bewertungslogik)
    *   [Phase 4: Der Durchbruch durch "Constrained Optimization"](#phase-4-der-durchbruch-durch-constrained-optimization)
    *   [Finale Testergebnisse](#finale-testergebnisse)
7.  [Glossar](#glossar)
8.  [Mitwirken](#mitwirken)
9.  [Lizenz](#lizenz)

## Über das Projekt

**Forge Studio** ist ein Framework zur Entdeckung und Optimierung von industriellen Rezepturen und neuen Materialien. Anstatt auf langwierige und teure Laborexperimente zu setzen, nutzt dieses Projekt einen datengesteuerten Ansatz, um den Suchraum potenzieller Materialkombinationen effizient zu durchsuchen.

Das Herzstück des Systems ist eine Kombination aus drei Schlüsseltechnologien:

1.  **GPU-Beschleunigung:** Ein maßgeschneiderter C-Treiber (`CipherCore_OpenCl.dll`), der über OpenCL direkt mit der Grafikkarte kommuniziert, um massive Parallelberechnungen durchzuführen. Dies beschleunigt den Optimierungsprozess um Größenordnungen im Vergleich zu reinen CPU-Implementierungen.
2.  **Surrogatmodellierung (MPG):** Ein "Mycelial Prototype Graph"-Modell wird auf existierenden Daten trainiert, um die Eigenschaften neuer, hypothetischer Materialien blitzschnell vorherzusagen, ohne dass eine teure Simulation oder ein Labortest erforderlich ist.
3.  **Evolutionäre Algorithmen:** Inspiriert von der biologischen Evolution, erzeugt und verbessert ein genetischer Algorithmus Generationen von "Rezepturen", um die besten Kandidaten zu finden, die den vom Benutzer definierten Zielen (z.B. hohe Stabilität, große Bandlücke) entsprechen.

## Kernkonzepte

### CipherCore OpenCL Treiber (DLL)

Die `CipherCore_OpenCl.dll` (oder `.so`/`.dylib` auf Linux/macOS) ist das Fundament der hohen Performance von Forge Studio. Diese in C geschriebene Bibliothek:

*   **Initialisiert die GPU:** Sie erkennt OpenCL-fähige Grafikkarten und stellt eine Verbindung her.
*   **Kompiliert Kernel:** Spezialisierte C-Programme (Kernel) für Aufgaben wie Matrixmultiplikation, Clustering und Simulation werden zur Laufzeit für die spezifische GPU-Architektur kompiliert.
*   **Verwaltet den Speicher:** Sie kümmert sich um das Zuweisen von Speicher auf der GPU und das schnelle Kopieren von Daten zwischen CPU und GPU.
*   **Stellt eine C-API bereit:** Python kommuniziert über das `ctypes`-Modul mit dieser Bibliothek, um Hochleistungsberechnungen anzustoßen.

### Mycelial Prototype Graph (MPG) Modell

Das MPG-Modell ist ein Surrogatmodell, das lernt, die Ergebnisse teurer Experimente oder Simulationen zu approximieren.

*   **Prototype Graph:** Das Modell lernt eine Reihe von "Prototypen" – ideale Repräsentanten für bestimmte Materialcluster (z.B. "stabile Oxide mit kleiner Dichte"). Wenn eine neue Rezeptur bewertet wird, wird sie mit diesen Prototypen verglichen. Ihre Eigenschaften werden dann aus einer gewichteten Kombination der Eigenschaften der nächstgelegenen Prototypen abgeleitet.
*   **Mycelial Guidance:** Der "myzeliale" Aspekt bezieht sich auf das Pheromon-System, das den evolutionären Algorithmus leitet. Wie ein Pilzmyzel, das Nährstoffe im Boden findet, hinterlässt der Algorithmus "Pheromonspuren" in den vielversprechenden Bereichen des Suchraums. Die `Guidance-Stärke` steuert, wie stark sich neue Generationen von diesen Spuren anziehen lassen.

### Evolutionärer Algorithmus

Der Optimierungsprozess ist ein genetischer Algorithmus:

1.  **Initialisierung:** Eine zufällige "Population" von Kandidaten-Rezepturen wird erstellt.
2.  **Bewertung (Fitness):** Das MPG-Modell sagt die Eigenschaften jedes Kandidaten voraus. Eine Bewertungsfunktion (`score_formulations`) berechnet einen "Fitness-Score" basierend auf den vom Benutzer gesetzten Zielen.
3.  **Selektion:** Die besten Kandidaten ("Eliten") werden für die nächste Generation ausgewählt.
4.  **Reproduktion:** Neue Kandidaten ("Kinder") werden durch `Crossover` (Kombination zweier Eltern) und `Mutation` (zufällige kleine Änderungen) erzeugt.
5.  **Wiederholung:** Der Prozess wird über Hunderte von Generationen wiederholt, wobei sich die Population schrittweise in Richtung optimaler Lösungen entwickelt.

## Architektur

Das Projekt ist in drei klare Schichten unterteilt:

1.  **Präsentationsschicht (`forge_studio_ui.py`):** Eine interaktive Web-Benutzeroberfläche, die mit Streamlit erstellt wurde. Sie ermöglicht die Konfiguration der Optimierung und die Visualisierung der Ergebnisse.
2.  **Logikschicht (`forge_backend.py`):** Das Python-Herzstück. Es implementiert den evolutionären Algorithmus, steuert den Trainingsprozess des MPG-Modells und fungiert als Brücke zur C-Bibliothek.
3.  **Compute-Schicht (`CipherCore_OpenCl.dll`):** Die in C geschriebene High-Performance-Bibliothek, die alle rechenintensiven Aufgaben auf der GPU ausführt.

```
[ UI (Streamlit) ] <--> [ Python Backend (Logik) ] <--> [ CipherCore DLL (C/OpenCL) ] <--> [ GPU ]
```

## Installation und Einrichtung

### Voraussetzungen

*   Python 3.10 oder höher
*   Ein C/C++ Compiler (z.B. GCC/Clang auf Linux, MSVC auf Windows)
*   Aktuelle Treiber für eine OpenCL 1.2+ fähige GPU (NVIDIA, AMD, Intel)
*   Git zur Versionskontrolle

### Schritt-für-Schritt-Anleitung

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Python-Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **CipherCore DLL kompilieren:**
    Navigieren Sie in das Verzeichnis, das den C-Quellcode (`opencl_driver.c`) enthält.

    *   **Unter Linux/macOS (mit GCC/Clang):**
        ```bash
        # Der -I Pfad muss möglicherweise an Ihr System angepasst werden
        gcc -shared -o CipherCore_OpenCl.so -fPIC opencl_driver.c -lOpenCL
        ```

    *   **Unter Windows (mit MSVC in einer Developer Command Prompt):**
        ```bash
        # 'path\to\opencl\sdk' anpassen
        cl /LD opencl_driver.c -I"path\to\opencl\sdk\include" "path\to\opencl\sdk\lib\OpenCL.lib" /FeCipherCore_OpenCl.dll
        ```    > **Wichtig:** Stellen Sie sicher, dass die kompilierte DLL (`CipherCore_OpenCl.dll` oder `.so`) im Hauptverzeichnis des Projekts liegt, damit sie von `autodetect_dll()` gefunden wird.

4.  **Anwendung starten:**
    ```bash
    streamlit run forge_studio_ui.py
    ```

## Anwendung und Nutzung

### Starten der Anwendung

Nach dem Start öffnet sich automatisch ein Browserfenster mit der Forge Studio UI unter `http://localhost:8501`.

### Konfiguration eines Optimierungslaufs

1.  **Problem auswählen:** Wählen Sie ein vordefiniertes Optimierungsproblem aus, z.B. "Stabile Halbleiter (Materials Project)".
2.  **Ziele definieren:** Wählen Sie die Materialeigenschaften aus, die Sie optimieren möchten (z.B. `Bandlücke`, `Bildungsenergie`). Weisen Sie Gewichte zu: positive Werte für Maximierung, negative für Minimierung.
3.  **Datenquelle konfigurieren:** Geben Sie für das Materials Project einen gültigen API-Schlüssel und eine chemische Systemabfrage (z.B. `*-O` für alle Oxide) an.
4.  **Simulations-Parameter einstellen:**
    *   **Population:** Anzahl der Kandidaten pro Generation (empfohlen: 512+).
    *   **Generationen:** Dauer der Suche (empfohlen: 100+).
    *   **Anzahl Prototypen:** Komplexität des MPG-Modells (empfohlen: 256+ für komplexe Probleme).
5.  **Myzel-Parameter anpassen:**
    *   **Guidance-Stärke:** Steuert die Balance zwischen Erkundung (niedrige Werte) und Ausnutzung (hohe Werte). Ein guter Startwert ist `0.1-0.2`.
6.  **Synthese starten:** Klicken Sie auf "Synthese starten", um den GPU-beschleunigten Lauf zu beginnen. Die Ergebnisse werden live in der Konsole und nach Abschluss in der UI angezeigt.

## Fallstudie: Von Fehlern zur erfolgreichen Optimierung

Die Entwicklung dieses Projekts war eine iterative Reise, die typische Herausforderungen bei der Kombination von Hardware-Beschleunigung und maschinellem Lernen widerspiegelt.

#### Phase 1: Technische Fehlerbehebung

Die ersten Hürden waren rein technischer Natur:
*   **DLL-Ladefehler:** Sicherstellen, dass die C-Bibliothek korrekt kompiliert und vom Python-Skript gefunden wird.
*   **Python-Fehler:** Behebung von `NameError`, `TypeError` und `SyntaxError` aufgrund fehlender Imports, falscher Funktionsaufrufe und Tippfehlern in der evolutionären Logik.

#### Phase 2: Das "kollabierte Modell"

Nachdem die technischen Fehler behoben waren, zeigten die ersten Läufe ein trügerisches Bild: Der Score sprang sofort auf 1.0, aber die Vorhersagen waren für alle Kandidaten identisch.
*   **Problem:** Das MPG-Modell war "unterangepasst". Es lernte nur den Durchschnittswert der Trainingsdaten und war nicht komplex genug, um auf Variationen in den Eingabedaten zu reagieren.
*   **Lösung:** Erhöhung der Trainingsdaten durch breitere Abfragen (`*-O`) und Hinzufügen aussagekräftigerer Features (`formation_energy_per_atom`).

#### Phase 3: Verfeinerung der Bewertungslogik

Die nächste Herausforderung war eine zu "flache" oder "steile" Bewertungslandschaft.
*   **Problem:** Eine harte "Straf-Klippe" in der `score_formulations`-Funktion (z.B. Score = -999 bei `e_above_hull > 0.1`) machte es dem Algorithmus unmöglich, sich schrittweise zu verbessern.
*   **Lösung:** Umwandlung der harten Strafen in "sanfte Rampen", die eine kontinuierliche Bewertung ermöglichen.

#### Phase 4: Der Durchbruch durch "Constrained Optimization"

Selbst mit sanften Strafen war das Problem für die KI noch zu komplex. Der entscheidende Durchbruch kam durch eine Vereinfachung der Aufgabe:
1.  **Daten filtern:** Das Modell wurde ausschließlich auf **stabilen Materialien** trainiert.
2.  **Ziel fokussieren:** Das Modell lernte, **nur die Bandlücke vorherzusagen**.
3.  **Score anpassen:** Die Bewertungsfunktion wurde so umgebaut, dass sie die vorhergesagte Bandlücke mit den "Genen" des Kandidaten (wie `formation_energy_per_atom`) kombiniert.

#### Finale Testergebnisse

Mit dieser finalen Architektur zeigt das System das gewünschte Verhalten einer echten Optimierung. Der Diagnostik-Tab zeigt einen **Fitness-Verlauf**, bei dem der Score niedrig beginnt und sich über die Generationen hinweg einem Optimum annähert. Dies beweist, dass der Algorithmus aktiv den Suchraum erkundet und lernt.


> *Finale Testergebnisse: Der "Fitness-Verlauf" zeigt, wie der beste Score (blaue Linie) über die Generationen ansteigt – ein klares Zeichen für eine erfolgreiche Optimierung.*

## Glossar

*   **API-Key:** Ein geheimer Schlüssel zur Authentifizierung bei einem Web-Dienst wie dem Materials Project.
*   **Bandlücke (Band Gap):** In der Festkörperphysik die Energie, die benötigt wird, um ein Elektron in einen leitenden Zustand anzuregen. Materialien mit einer Bandlücke sind Halbleiter oder Isolatoren.
*   **DLL (Dynamic Link Library):** Eine unter Windows kompilierte Bibliothek mit C-Funktionen, die von anderen Programmen (hier: Python) aufgerufen werden kann.
*   **Energie über der Hull (Energy Above Hull):** Ein Maß für die thermodynamische Stabilität eines Materials. Ein Wert von 0 eV/Atom bedeutet, dass das Material stabil ist. Positive Werte deuten auf Instabilität hin.
*   **Evolutionärer Algorithmus:** Eine Optimierungstechnik, die von der biologischen Evolution inspiriert ist (Selektion, Kreuzung, Mutation).
*   **Feature:** Eine einzelne messbare Eigenschaft oder ein Merkmal, das als Eingabe für ein Machine-Learning-Modell dient (z.B. `density`).
*   **Kernel (OpenCL):** Ein kleines, in C geschriebenes Programm, das zur Ausführung auf den vielen Recheneinheiten einer GPU konzipiert ist.
*   **MPG (Mycelial Prototype Graph):** Der Name des in diesem Projekt verwendeten Surrogatmodells.
*   **OpenCL (Open Computing Language):** Ein offener Standard zur Programmierung von heterogenen Systemen, insbesondere zur Ausführung von Code auf GPUs.
*   **Pheromon:** Im Kontext dieses Projekts eine Metapher für eine "Erinnerungsspur", die der Algorithmus in vielversprechenden Bereichen des Suchraums hinterlässt.
*   **Prototyp:** Ein repräsentativer Datenpunkt, der einen Cluster oder eine "Region" im Datenraum repräsentiert. Das MPG-Modell nutzt eine Reihe von Prototypen, um sein "Wissen" zu strukturieren.
*   **Streamlit:** Ein Python-Framework zur schnellen Erstellung von interaktiven Web-Anwendungen für Data-Science-Projekte.
*   **Surrogatmodell:** Ein Machine-Learning-Modell, das als schneller "Ersatz" (Surrogat) für eine langsame oder teure Funktion (z.B. eine physikalische Simulation) dient.

## Mitwirken

Beiträge sind willkommen! Bitte öffnen Sie ein Issue, um Fehler zu melden oder neue Features vorzuschlagen.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der `LICENSE`-Datei.