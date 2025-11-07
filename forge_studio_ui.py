#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forge_studio_ui.py (Version 5.0 - Chemical Feature Engineering)
===============================================================
- Passt die ColumnMap an, um 'chemsys' als Feature anzufordern.
- Ermöglicht dem Backend, chemische Informationen für das Modelltraining zu nutzen.
- Behält den vollständig integrierten Workflow von Optimierung bis Rezept bei.
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
import recipe_mapper as mapper

# ===================================================================
# ===  PROBLEM-DEFINITIONEN                                       ===
# ===================================================================

try:
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists("data/concrete_data.csv"):
        st.warning("Datei 'data/concrete_data.csv' nicht gefunden. Erstelle eine leere Dummy-Datei.")
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
    "Stabile Halbleiter (Materials Project)": {
        "data_source": {"chemsys": "Si-O"},
        "search_space": {
            "density": (2.0, 8.0), 
            "volume": (20.0, 500.0), 
            "nsites": (2, 24),
            "formation_energy_per_atom": (-8.0, 0.5)
        },
        "objectives": {
            "band_gap": "Bandlücke (eV)", 
            "e_above_hull": "Energie über der Hull (eV/Atom)",
            "formation_energy_per_atom": "Bildungsenergie (eV/Atom)"
        },
        "presets": {
            "Stabiler Halbleiter": {"e_above_hull": -1.0, "band_gap": 1.0},
            "Breite Bandlücke": {"band_gap": 1.0, "e_above_hull": -0.5},
        },
        "source_type": "Materials Project",
        "column_map": ColumnMap(
            features={
                "density": "density", 
                "volume": "volume", 
                "nsites": "nsites",
                "formation_energy_per_atom": "formation_energy_per_atom",
                "chemsys": "chemsys"  # HINZUGEFÜGT: "Chemische Brille"
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
            if cols[i % len(cols)].button(f"Preset: {pname}", key=f"{state_key}_preset_{pname}"):
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

tabs = ["B) Neue Formulierung entwerfen", "C) Diagnostik"]
if "df_formulations" in st.session_state and st.session_state.df_formulations is not None:
    tabs.append("D) Rezept-Generator")

tab_widgets = st.tabs(tabs)
tabB = tab_widgets[0]
tabC = tab_widgets[1]
tabD = None
if len(tab_widgets) > 2:
    tabD = tab_widgets[2]

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
        mp_api_key = st.text_input("Materials Project API-Key", type="password", help="Wird nicht gespeichert. Alternativ MP_API_KEY als Umgebungsvariable setzen.", key="mp_api_key_input")
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
    pop_B = c1.number_input("Population", 32, 4096, 512, 16)
    steps_B = c2.number_input("Generationen", 10, 2000, 100, 10)
    n_prototypes_B = c3_proto.number_input("Anzahl Prototypen (Modellkomplexität)", 32, 2048, 512, 32)

    st.markdown("---")
    st.subheader("🧠 Myzel-Parameter für die Suche")
    c1, c2, c3 = st.columns(3)
    guidance = c1.slider("Guidance-Stärke", 0.0, 1.0, 0.2, 0.05)
    decay = c2.slider("Zerfall (Decay)", 0.0, 0.5, 0.1, 0.01)
    diffusion = c3.slider("Diffusion", 0.0, 0.2, 0.05, 0.01)
    k_neighbors = st.number_input("k-Nachbarn (Topologie)", 1, 16, 8, 1)

    if st.button("🧬 Synthese starten", type="primary"):
        if not objectives_B: st.error("Bitte mindestens ein Ziel definieren.")
        elif problem_def["source_type"] == "Materials Project" and not (mp_api_key or os.getenv("MP_API_KEY")):
            st.error("Für eine Abfrage an das Materials Project wird ein API-Key benötigt.")
        else:
            st.session_state['start_time'] = time.perf_counter()
            with st.spinner("Evolution startet... GPU wird initialisiert und Modell wird trainiert..."):
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
                        n_prototypes=int(n_prototypes_B),
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
                    st.error(f"Ein Wertfehler ist aufgetreten: {e}")
                except Exception as e:
                    st.exception(e)

    if "df_formulations" in st.session_state and st.session_state.df_formulations is not None:
        st.subheader("Vorgeschlagene Rezepturen")
        df_display = st.session_state.df_formulations
        st.dataframe(df_display)
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
        st.metric(label=f"Letzter Lauf: {last.get('mode','')}", value=last.get('backend_fmt','n/a'), help=f"Beendet am: {last.get('timestamp_end','')}")
    else: st.info("Noch keine Laufzeit erfasst. Starte eine Synthese in Tab B.")

    if 'meta_formulations' in st.session_state and st.session_state.meta_formulations:
        meta = st.session_state.meta_formulations
        gen_hist = meta.get("gen_history")
        if gen_hist:
            st.subheader("📈 Generationsmetriken")
            gh = pd.DataFrame(gen_hist)
            c1, c2 = st.columns(2)
            c1.write("**Fitness-Verlauf**")
            c1.line_chart(gh.set_index("gen")[["best_norm", "mean_norm"]], height=300)
            c2.write("**Myzel-Aktivität (Pheromon-Durchschnitt)**")
            c2.line_chart(gh.set_index("gen")["pheromone_mean"], height=300)
            with st.expander("Rohdaten der Generationen (gen_history)"): st.dataframe(gh)
    else: st.info("Keine Metriken gefunden. Starte eine Synthese in Tab B.")

if tabD:
    with tabD:
        st.header("D) Konkrete Kandidaten & Labor-Rezepte")
        st.info("Dieses Modul sucht in der Materials Project Datenbank nach realen Materialien, die den optimierten Eigenschaften am nächsten kommen.")

        results_df = st.session_state.df_formulations
        temp_csv_path = "temp_results_for_mapper.csv"
        results_df.to_csv(temp_csv_path, index=False)

        top_k_candidates = st.number_input("Top-K Kandidaten pro Zielzeile", min_value=1, max_value=10, value=3, step=1, key="mapper_top_k")
        
        st.subheader("🧪 Batch-Rezept-Generator")
        make_recipes = st.checkbox("Rezepte für Top-Konsens-Kandidaten erstellen")
        
        recipe_mass = None
        recipe_top_n = None
        if make_recipes:
            rc1, rc2 = st.columns(2)
            recipe_mass = rc1.number_input("Ziel-Batchmasse (g)", min_value=1.0, value=50.0, step=10.0)
            recipe_top_n = rc2.number_input("Rezepte für Top-N Konsens-Kandidaten", min_value=1, max_value=20, value=5, step=1)

        if st.button("🗺️ Kandidaten-Mapping starten", type="primary"):
            # Holt den API-Schlüssel erneut, da der Zustand zwischen den Läufen verloren gehen kann
            api_key_from_session = st.session_state.get("mp_api_key_input")
            api_key_for_mapper = api_key_from_session or os.getenv("MP_API_KEY")
            if not api_key_for_mapper:
                st.error("Ein Materials Project API-Key wird für das Mapping benötigt. Bitte geben Sie ihn in Tab B an oder setzen Sie die Umgebungsvariable.")
            else:
                os.environ['MP_API_KEY'] = api_key_for_mapper # Setze temporär für den Mapper-Prozess
                with st.spinner("Suche nach realen Materialkandidaten..."):
                    try:
                        mapped_results = mapper.map_candidates(temp_csv_path, topk=int(top_k_candidates))
                        st.session_state.mapped_results = mapped_results
                        
                        targets_df = pd.read_csv(temp_csv_path)
                        df_cons = mapper._consensus_build(mapped_results, targets_df)
                        st.session_state.consensus_df = df_cons
                        
                        if make_recipes and recipe_mass and recipe_top_n:
                            consensus_temp_path = "temp_consensus.csv"
                            df_cons.to_csv(consensus_temp_path, index=False)
                            recipes = mapper.make_recipes_from_consensus(consensus_temp_path, topn=int(recipe_top_n), target_mass_g=float(recipe_mass))
                            st.session_state.recipes = recipes
                            if os.path.exists(consensus_temp_path): os.remove(consensus_temp_path)

                        st.success("Mapping und Analyse abgeschlossen!")
                    
                    except Exception as e:
                        st.exception(e)
        
        if "mapped_results" in st.session_state:
            st.markdown("---")
            st.subheader("Vorgeschlagene reale Materialien")
            
            for i, (target_row, candidate_list) in enumerate(zip(results_df.head(len(st.session_state.mapped_results)).to_dict('records'), st.session_state.mapped_results)):
                with st.expander(f"**Ziel #{i+1}:** (pred_Eg={target_row['pred_band_gap']:.2f} eV) → **{len(candidate_list)} Kandidaten gefunden**"):
                    if not candidate_list:
                        st.write("Keine passenden Materialien in der Datenbank gefunden.")
                        continue
                    
                    for j, c in enumerate(candidate_list):
                        hull = "n/a" if c.energy_above_hull is None else f"{c.energy_above_hull:.3f} eV/Atom"
                        rho = "n/a" if c.density is None else f"{c.density:.2f} g/cm³"
                        Eg = "n/a" if c.band_gap is None else f"{c.band_gap:.2f} eV"
                        
                        st.markdown(f"**{j+1}. {c.formula}** (`{c.material_id}`)")
                        st.markdown(f"> E-Hull: **{hull}** | Dichte: **{rho}** | Bandlücke: **{Eg}**")
                        st.caption(f"Synthese-Hinweis: {c.notes}")
        
        if "consensus_df" in st.session_state:
            st.markdown("---")
            st.subheader("Konsens-Ranking")
            st.write("Die besten Kandidaten, die über alle Ziel-Rezepturen hinweg am häufigsten und mit der geringsten Abweichung vorgeschlagen wurden.")
            st.dataframe(st.session_state.consensus_df)
        
        if "recipes" in st.session_state:
            st.markdown("---")
            st.subheader("Generierte Labor-Rezepte")
            
            for r in st.session_state.recipes:
                st.markdown(f"#### Rezept für: **{r.formula}** (`{r.material_id}`)")
                st.write(f"**Zielmenge:** {r.target_mass_g:.2f} g | **Synthese-Route:** {r.route}")
                
                prec_df = pd.DataFrame(r.precursors)
                st.table(prec_df[['name', 'formula', 'mass_g', 'mol']])
                
                with st.expander("Prozess-Schritte (Checkliste)"):
                    for note in r.notes:
                        st.markdown(f"- {note}")

        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)