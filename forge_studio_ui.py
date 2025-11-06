#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forge_studio_ui.py (Version 3.3 - Final)
===========================================
- Behebt `KeyError` durch bedingte Anzeige der Query-Box.
- Behält alle UI-Verbesserungen wie File-Uploader und CSV-Download bei.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import time
from datetime import datetime
import os
import numpy as np

import forge_backend as forge
from materials_mp_connector import ColumnMap

# ===================================================================
# ===  PROBLEM-DEFINITIONEN                                       ===
# ===================================================================

try:
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists("data/concrete_data.csv"):
        st.warning("Datei 'data/concrete_data.csv' nicht gefunden. Erstelle eine leere Dummy-Datei. Für beste Ergebnisse laden Sie bitte den UCI Concrete Datensatz herunter und platzieren ihn in 'data/'.")
        pd.DataFrame(columns=["Zement", "Schlacke", "Flugasche", "Wasser", "Superplastifizierer", "Grober_Zuschlag", "Feiner_Zuschlag", "Alter", "Druckfestigkeit"]).to_csv("data/concrete_data.csv", index=False)
    
    df_concrete_check = pd.read_csv("data/concrete_data.csv")
    if df_concrete_check.empty: raise FileNotFoundError
    search_space_concrete = {col: (float(df_concrete_check[col].min()), float(df_concrete_check[col].max())) for col in df_concrete_check.columns if col != "Druckfestigkeit"}
except (FileNotFoundError, Exception):
    search_space_concrete = {
        "Zement": (102.0, 540.0), "Schlacke": (0.0, 359.4), "Flugasche": (0.0, 200.1),
        "Wasser": (121.8, 247.0), "Superplastifizierer": (0.0, 32.2),
        "Grober_Zuschlag": (801.0, 1145.0), "Feiner_Zuschlag": (594.0, 992.6), "Alter": (1, 365)
    }

PROBLEM_DEFINITIONS = {
    "Hochleistungsbeton (Lokale CSV)": {
        "data_source": "data/concrete_data.csv",
        "search_space": search_space_concrete,
        "objectives": {"Druckfestigkeit": "Druckfestigkeit (MPa)"},
        "presets": {"Maximale Festigkeit": {"Druckfestigkeit": 1.0}},
        "source_type": "Lokale CSV"
    },
    # NEUER CODE
    "Stabile Halbleiter (Materials Project)": {
        "data_source": {"chemsys": "Si-Ge-O"},
        # WICHTIG: Neues Feature zum Suchraum hinzufügen
        "search_space": {
            "density": (2.0, 8.0), 
            "volume": (20.0, 500.0), 
            "nsites": (2, 24),
            "formation_energy_per_atom": (-4.0, 0.5) # Suchbereich für die Bildungsenergie
        },
        "objectives": {
            "band_gap": "Bandlücke (eV)", 
            "e_above_hull": "Energie über der Hull (eV/Atom)",
            "formation_energy_per_atom": "Bildungsenergie (eV/Atom)" # <-- HIER HINZUFÜGEN
        },
        "presets": {
            "Stabiler Halbleiter": {"e_above_hull": -1.0, "band_gap": 1.0},
            "Breite Bandlücke": {"band_gap": 1.0, "e_above_hull": -0.5},
        },
        "source_type": "Materials Project",
        "column_map": ColumnMap(
            # WICHTIG: Neues Feature zur Feature-Map hinzufügen
            features={
                "density": "density", 
                "volume": "volume", 
                "nsites": "nsites",
                "formation_energy_per_atom": "formation_energy_per_atom"
            },
            targets={"band_gap": "band_gap", "e_above_hull": "energy_above_hull"}
        )
    }
}

# ===================================================================
# ===  UI-HELFERFUNKTIONEN                                        ===
# ===================================================================

def objective_builder(title: str, available: Dict, *, state_key: str, presets: Dict | None = None) -> Dict[str, float]:
    st.subheader(title)
    if state_key not in st.session_state: st.session_state[state_key] = {}

    if presets:
        cols = st.columns(min(4, len(presets)))
        for i, (pname, pobj) in enumerate(presets.items()):
            if cols[i % len(cols)].button(f"Preset: {pname}", use_container_width=True, key=f"{state_key}_preset_{pname}"):
                st.session_state[state_key] = dict(pobj); st.rerun()
    
    current_objectives = st.session_state.get(state_key, {})
    selected = st.multiselect("Ziele auswählen", options=list(available.keys()), format_func=lambda k: available.get(k, k), default=list(current_objectives.keys()), key=f"{state_key}_select")
    
    new_objectives = {k: v for k, v in current_objectives.items() if k in selected}
    for k in selected:
        if k not in new_objectives: new_objectives[k] = -1.0 if any(s in k.lower() for s in ["kosten", "energy", "hull"]) else 1.0
    st.session_state[state_key] = new_objectives

    if selected:
        st.markdown("**Gewichte je Ziel** (`>0` = maximieren, `<0` = minimieren)")
        for k in selected:
            new_objectives[k] = st.number_input(f"Gewicht für '{available[k]}'", key=f"{state_key}_w_{k}", value=float(new_objectives[k]), step=0.1, format="%.2f")
        st.session_state[state_key] = new_objectives
    else:
        st.info("Bitte wählen Sie mindestens ein Optimierungsziel aus.")
    
    return st.session_state.get(state_key, {})

def _fmt_duration(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds): return "n/a"
    ms = int(round((seconds - int(seconds)) * 1000))
    s, m, h = int(seconds) % 60, int(seconds) // 60 % 60, int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# ===================================================================
# ===  HAUPT-LAYOUT DER ANWENDUNG                                   ===
# ===================================================================

st.set_page_config(page_title="Forge Studio: Formulation Engine", layout="wide")
st.title("🔬 Forge Studio: Industrial Formulation Engine")
st.caption("GPU-beschleunigte Rezepturoptimierung mit dem Mycelial Prototype Graph (MPG) Modell")

with st.sidebar:
    st.header("⚙️ Globale Einstellungen")
    dll_path = st.text_input("DLL-Pfad", value=forge.autodetect_dll())
    gpu_index = st.number_input("GPU-Index", min_value=0, value=0, step=1)

tabB, tabC = st.tabs(["B) Neue Formulierung entwerfen", "C) Diagnostik"])

with tabB:
    st.header("B) Evolutionäre Formulierungs-Optimierung")
    
    problem_name = st.selectbox("Anwendungsproblem auswählen:", list(PROBLEM_DEFINITIONS.keys()), key="problem_selector")
    problem_def = PROBLEM_DEFINITIONS[problem_name]
    
    objective_state_key = f"obj_builder_{problem_name.replace(' ', '_')}"
    objectives_B = objective_builder("Ziele definieren", available=problem_def["objectives"], state_key=objective_state_key, presets=problem_def.get("presets"))
    
    st.markdown("---")
    st.subheader("Datenquelle & Simulations-Parameter")

    mp_api_key = None
    data_source_arg = None
    if problem_def["source_type"] == "Materials Project":
        st.info("Dieses Problem nutzt Live-Daten vom Materials Project. API-Key wird benötigt.")
        mp_api_key = st.text_input("Materials Project API-Key", type="password", help="Wird nicht gespeichert. Alternativ MP_API_KEY als Umgebungsvariable setzen.")
        # *** KORREKTUR: Sicherer Zugriff auf 'chemsys' ***
        default_query = problem_def["data_source"].get("chemsys", "Si-O")
        query_str = st.text_input("MP Query (chem. System, z.B. 'Si-Ge-O')", value=default_query)
        data_source_arg = {"chemsys": query_str}
    else:
        uploaded_file = st.file_uploader("Eigene Trainingsdaten hochladen (optional, CSV)", type=['csv'])
        if uploaded_file is not None:
            data_source_arg = uploaded_file
        else:
            data_source_arg = st.text_input("Pfad zu den Trainingsdaten (CSV)", problem_def["data_source"])

    c1, c2, c3_proto = st.columns(3)
    pop_B = c1.number_input("Population", 32, 2048, 512, 16) # Standard auf 512 gesetzt
    steps_B = c2.number_input("Generationen", 10, 1000, 200, 10) # Standard auf 200 gesetzt
    n_prototypes_B = c3_proto.number_input("Anzahl Prototypen (Modellkomplexität)", 32, 1024, 64, 32)

    st.markdown("---")
    st.subheader("🧠 Myzel-Parameter für die Suche")
    c1, c2, c3 = st.columns(3)
    guidance = c1.slider("Guidance-Stärke", 0.0, 1.0, 0.4, 0.05)
    decay = c2.slider("Zerfall (Decay)", 0.0, 0.5, 0.1, 0.01)
    diffusion = c3.slider("Diffusion", 0.0, 0.2, 0.05, 0.01)
    k_neighbors = st.number_input("k-Nachbarn (Topologie)", 1, 16, 8, 1)

    if st.button("🧬 Synthese starten", type="primary", use_container_width=True):
        if not objectives_B: st.error("Bitte mindestens ein Ziel definieren.")
        elif problem_def["source_type"] == "Materials Project" and not (mp_api_key or os.getenv("MP_API_KEY")):
            st.error("Für eine Abfrage an das Materials Project wird ein API-Key benötigt.")
        else:
            st.session_state['start_time'] = time.perf_counter()
            with st.spinner("Evolution startet..."):
                try:
                    result_df, meta = forge.mycelial_guided_evolution(
                                            dll_path=dll_path, 
                                            gpu_index=int(gpu_index),
                                            source_type=problem_def["source_type"], 
                                            data_source_arg=data_source_arg,
                                            mp_api_key=mp_api_key, 
                                            mp_column_map=problem_def.get("column_map"),
                                            search_space=problem_def["search_space"], 
                                            objectives=objectives_B,
                                            population=int(pop_B), 
                                            steps=int(steps_B),
                                            n_prototypes=int(n_prototypes_B),  # <-- HIER IST DIE NEUE ZEILE
                                            mycel_guidance_strength=float(guidance), 
                                            mycel_decay=float(decay),
                                            mycel_diffusion=float(diffusion), 
                                            mycel_k_neighbors=int(k_neighbors),
                                        )
                    
                    t_backend = float(meta.get("runtime_s", 0.0))
                    
                    st.session_state.df_formulations = result_df
                    st.session_state.meta_formulations = meta
                    st.session_state.last_run_timing = {
                        "mode": f"Suche für '{problem_name}'", "backend_fmt": _fmt_duration(t_backend),
                        "timestamp_end": datetime.now().astimezone().isoformat(timespec="seconds"),
                    }
                    st.success("Synthese abgeschlossen!")
                    st.rerun()
                
                except ValueError as e:
                    if "Trainingsdaten sind nach der Bereinigung leer" in str(e):
                        st.error(f"Fehler: {e} Ihre Abfrage lieferte keine Daten, die alle benötigten Spalten enthalten. Versuchen Sie eine breitere Abfrage (z.B. 'Si-O' statt 'Si-Ge-O').")
                    else: st.exception(e)
                except Exception as e:
                    st.exception(e)

    if "df_formulations" in st.session_state and st.session_state.df_formulations is not None:
        st.subheader("Vorgeschlagene Rezepturen")
        df_display = st.session_state.df_formulations
        st.dataframe(df_display, use_container_width=True)
        st.download_button(
            "Als CSV herunterladen",
            df_display.to_csv(index=False).encode("utf-8"),
            file_name=f"forge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with tabC:
    st.header("C) Diagnostik der evolutionären Suche")
    if 'last_run_timing' in st.session_state:
        last = st.session_state.last_run_timing
        st.subheader("⏱️ Letzte Laufzeit")
        st.write(f"- Problem: **{last.get('mode','')}**\n- Backend-Kern: **{last.get('backend_fmt','n/a')}**\n- Fertig um: {last.get('timestamp_end','')}")
    else: st.info("Noch keine Laufzeit erfasst. Starte eine Synthese in Tab B.")

    if 'meta_formulations' in st.session_state and st.session_state.meta_formulations:
        meta = st.session_state.meta_formulations
        gen_hist = meta.get("gen_history")
        if gen_hist:
            st.subheader("📈 Generationsmetriken")
            gh = pd.DataFrame(gen_hist)
            c1, c2 = st.columns(2)
            c1.write("**Fitness-Verlauf**"); c1.line_chart(gh.set_index("gen")[["best_norm", "mean_norm"]], height=300)
            c2.write("**Myzel-Aktivität**"); c2.line_chart(gh.set_index("gen")["pheromone_mean"], height=300)
            with st.expander("Rohdaten der Generationen (gen_history)"): st.dataframe(gh)
    else: st.info("Keine Metriken gefunden. Starte eine Synthese in Tab B.")