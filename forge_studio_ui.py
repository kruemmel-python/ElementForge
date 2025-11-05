#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forge_studio_ui.py
==================
UI für:
A) Element-Bewertung (klassisch + Quantum/VQE)
B) Material-Synthese (Evolution) mit *echter* Myzel-Guidance
   + optionaler VQE-Einbindung in die Fitness (Top-Eliten pro Generation)
C) Diagnostik (GA-, Myzel- und Surrogat-Health)

Start:
  streamlit run forge_studio_ui.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import time
from datetime import datetime
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message=r".*multiple allotropes.*", category=UserWarning)

import forge_backend as forge

# -------------------------------------------------------------------
# „freundliche“ Properties je Kontext (Key = Backend-Name)
# -------------------------------------------------------------------
ELEMENT_PROPS: Dict[str, str] = {
    "density": "Dichte (density)",
    "melting_point": "Schmelzpunkt (melting_point)",
    "boiling_point": "Siedepunkt (boiling_point)",
    "atomic_weight": "Atomgewicht (atomic_weight)",
    "electronegativity_pauling": "Elektronegativität (Pauling)",
    "vdw_radius": "van-der-Waals-Radius (vdw_radius)",
    "covalent_radius": "Kovalenzradius (covalent_radius)",
    "ionization_energy": "Ionisationsenergie (1. Stufe)",
}

MATERIAL_PROPS: Dict[str, str] = {
    "bandgap": "Bandlücke (bandgap)",
    "formation_energy": "Bildungsenergie/Atom (formation_energy)",
    "density": "Dichte (density)",
    # ggf. erweitern, wenn im Datensatz vorhanden:
    # "bulk_modulus": "Bulkmodul (bulk_modulus)",
    # "shear_modulus": "Schermodul (shear_modulus)",
}

# -------------------------------------------------------------------
# Ziel-Builder: klickbare Auswahl + Gewichte
# -------------------------------------------------------------------
def objective_builder(
    title: str,
    available: Dict[str, str],
    *,
    state_key: str,
    presets: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, float]:
    st.subheader(title)

    # Session-Init
    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    # Presets
    if presets:
        cols = st.columns(min(4, len(presets)))
        for i, (pname, pobj) in enumerate(presets.items()):
            if cols[i % len(cols)].button(
                f"Preset: {pname}",
                use_container_width=True,
                key=f"{state_key}_preset_{pname}",
            ):
                st.session_state[state_key] = dict(pobj)

    # Auswahl
    current_keys: List[str] = list(st.session_state[state_key].keys())
    selected = st.multiselect(
        "Ziele auswählen",
        options=list(available.keys()),
        format_func=lambda k: available[k],
        default=current_keys or None,
        help="Positive Gewichte = maximieren, negative = minimieren.",
        key=f"{state_key}_select",
    )

    # Entfernte Keys räumen
    for k in list(st.session_state[state_key].keys()):
        if k not in selected:
            st.session_state[state_key].pop(k, None)

    # Neu hinzugefügte Keys initialisieren
    for k in selected:
        if k not in st.session_state[state_key]:
            st.session_state[state_key][k] = float(-1.0 if k in {"formation_energy"} else 1.0)

    # Gewichte editieren
    if selected:
        st.markdown("**Gewichte je Ziel**  \n(>0 = maximieren, <0 = minimieren; Mischung möglich)")
        for k in selected:
            label = available[k]
            cols = st.columns([2, 2, 1])
            with cols[0]:
                st.write(f"• {label}")
            with cols[1]:
                w = st.number_input(
                    "Gewicht",
                    key=f"{state_key}_w_{k}",
                    value=float(st.session_state[state_key][k]),
                    step=0.1,
                    format="%.2f",
                )
                st.session_state[state_key][k] = float(w)
            with cols[2]:
                if st.button("➖ Entfernen", key=f"{state_key}_rm_{k}"):
                    st.session_state[state_key].pop(k, None)
                    st.rerun()
    else:
        st.info("Noch keine Ziele gewählt. Bitte oben Eigenschaften hinzufügen.")

    return dict(st.session_state[state_key])

# --- Zeitformatierer ---
def _fmt_duration(seconds: float) -> str:
    """Formatiert Sekunden als HH:MM:SS.mmm (millisekundengenau)."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# -------------------------------------------------------------------
# Datenzugriff / Caching – nur hashbare Inputs verwenden!
# -------------------------------------------------------------------

def _load_df(dataset: str) -> pd.DataFrame:
    """Nicht cachen: der Original-DataFrame enthält teils dict-Spalten (nicht hashbar)."""
    return forge.load_jarvis_dataframe(dataset)

@st.cache_data(show_spinner=False)
def _cached_vocab_from_formulas(formulas: Tuple[str, ...], max_elems: int) -> list[str]:
    """
    Erzeugt das Vokabular nur aus Formeln (hashbar; keine dict-Spalten).
    """
    # Minimal-DataFrame bauen, damit wir forge.build_vocab wiederverwenden können
    df_small = pd.DataFrame({"formula": list(formulas)})
    return forge.build_vocab(df_small, max_elems=max_elems)

@st.cache_data(show_spinner=False)
def _cached_surrogates_from_targets(df_targets: pd.DataFrame, vocab: list[str], obj_list: list[str]) -> dict[str, forge.Surrogate]:
    """
    Cacht die Surrogate auf einem *schlanken* DataFrame (nur 'formula' + float-Zielspalten).
    Solche DFs sind hashbar (keine dict-Objekte in Zellen).
    """
    # Sicherheitsnetz: nur 'formula' + obj_list behalten, cast auf numerisch
    cols = ["formula"] + list(obj_list)
    df_slim = df_targets.loc[:, [c for c in cols if c in df_targets.columns]].copy()
    for obj in obj_list:
        if obj in df_slim.columns:
            df_slim[obj] = pd.to_numeric(df_slim[obj], errors="coerce")
    return forge.safe_train_surrogate_map(df_slim, vocab, obj_list)

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
st.set_page_config(page_title="Forge Studio", layout="wide")
st.title("🌌 Forge Studio: Elements & Materials")
st.caption("GPU-beschleunigt via CipherCore DLL mit Myzel- & VQE-Fitness")

with st.sidebar:
    st.header("⚙️ Globale Einstellungen")
    dll_path = st.text_input("DLL-Pfad", value=forge.autodetect_dll())
    gpu_index = st.number_input("GPU-Index", min_value=0, value=0, step=1)

tabA, tabB, tabC = st.tabs([
    "A) Elemente bewerten",
    "B) Neues Material entwerfen",
    "C) Diagnostik",
])

# ===================================================================
# TAB A – Elemente
# ===================================================================
with tabA:
    st.header("A) Bewertung realer Elemente")
    st.markdown("Wähle die **Eigenschaften** und setze **Gewichte**: _positiv = maximieren_, _negativ = minimieren_.")

    element_presets = {
        "Leicht & schmelzstark": {"density": -1.0, "melting_point": 1.0},
        "Reaktiv & leicht": {"electronegativity_pauling": 1.0, "density": -0.5},
        "Hitzebeständig": {"melting_point": 1.0, "boiling_point": 0.5},
    }
    objectives_A = objective_builder("Ziele (Eigenschaft: Gewicht)", ELEMENT_PROPS, state_key="obj_builder_A", presets=element_presets)

    mode_A = st.radio("Berechnungsmodus", ["Klassisch (Einzigartigkeit)", "Quantum (VQE)"], horizontal=True, key="mode_A")

    if "Klassisch" in mode_A:
        uniq_w = st.slider("Gewichtung Einzigartigkeit (α)", 0.0, 1.0, 0.3, 0.05, key="uniq_w")
    else:
        c1, c2 = st.columns(2)
        with c1:
            quant_w = st.slider("Gewichtung Quantum-Score (β)", 0.0, 1.0, 0.5, 0.05, key="quant_w")
        with c2:
            qbits = st.number_input("Qubits (VQE)", min_value=2, max_value=16, value=4, step=1, key="qbits_A")

    if st.button("🚀 Elemente bewerten", type="primary"):
        if not objectives_A:
            st.error("Bitte mindestens ein Ziel definieren.")
        else:
            mode_key = "quantum" if "Quantum" in mode_A else "classic"
            with st.spinner(f"Bewertung läuft ({mode_key}) …"):
                try:
                    dfA = forge.score_elements(
                        dll_path=dll_path,
                        gpu_index=int(gpu_index),
                        mode=mode_key,
                        objectives=objectives_A,
                        uniqueness_weight=uniq_w if mode_key == "classic" else 0.3,
                        quantum_weight=quant_w if mode_key == "quantum" else 0.5,
                        num_qubits=qbits if mode_key == "quantum" else 4,
                    )
                    st.session_state["df_elements"] = dfA
                    st.success("Bewertung abgeschlossen.")
                except Exception as e:
                    st.exception(e)

    if "df_elements" in st.session_state:
        st.subheader("Rangliste")
        st.dataframe(st.session_state["df_elements"], width="stretch")
        st.download_button(
            "⤓ CSV exportieren",
            data=st.session_state["df_elements"].to_csv(index=False).encode("utf-8"),
            file_name="element_scores.csv",
            mime="text/csv",
        )

# ===================================================================
# TAB B – Material-Synthese (Myzel + VQE)
# ===================================================================
with tabB:
    st.header("B) Evolutionäre Material-Synthese (Myzel + VQE-Fitness optional)")
    st.markdown("Wähle **Materialziele**. Gewichte: _positiv = maximieren_, _negativ = minimieren_.")

    material_presets = {
        "Halbleiter-Fokus": {"bandgap": 1.0, "formation_energy": -1.0, "density": 0.3},
        "Strukturell leicht": {"density": -1.0, "formation_energy": -0.5},
        "Energie-stabil": {"formation_energy": -1.0},
    }
    objectives_B = objective_builder("Ziele (Eigenschaft: Gewicht)", MATERIAL_PROPS, state_key="obj_builder_B", presets=material_presets)

    c1, c2, c3 = st.columns(3)
    with c1:
        dataset_B = st.text_input("JARVIS-Datensatz", "dft_3d")
    with c2:
        pop_B = st.number_input("Population", min_value=32, max_value=1024, value=128, step=16)
    with c3:
        steps_B = st.number_input("Generationen", min_value=10, max_value=1000, value=50, step=10)

    c4, c5, c6 = st.columns(3)
    with c4:
        vocab_B = st.number_input("Vokabulargröße", min_value=16, max_value=128, value=32, step=4)
    with c5:
        max_elems_B = st.number_input("Max. Elemente / Formel", min_value=2, max_value=8, value=4, step=1)
    with c6:
        qbits_B = st.number_input("Qubits (Platzhalter für spätere tiefe VQE-Scorer)", min_value=2, max_value=16, value=5, step=1)

    # Persistiere aktuelle Auswahl für Diagnostik
    st.session_state["dataset_B"] = dataset_B
    st.session_state["vocab_B"] = int(vocab_B)

    st.markdown("---")
    st.subheader("🧠 Myzel-Parameter")
    use_mycel = st.checkbox("Myzel-Guidance aktivieren (CipherCore)", value=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        guidance = st.slider("Guidance-Stärke", 0.0, 1.0, 0.3, 0.05)
    with c8:
        decay = st.slider("Zerfall (Decay)", 0.0, 0.2, 0.05, 0.01)
    with c9:
        diffusion = st.slider("Diffusion", 0.0, 0.1, 0.02, 0.01)

    c10, c11 = st.columns(2)
    with c10:
        k_neighbors = st.number_input("k-Nachbarn (Topologie)", min_value=1, max_value=8, value=4, step=1)
    with c11:
        topk_bias = st.number_input("Top-k Bias (Pheromon Fokus, 0=aus)", min_value=0, max_value=64, value=0, step=1)

    st.markdown("---")
    st.subheader("⚛️ VQE direkt in die Fitness einmischen (GPU)")
    use_vqe_fit = st.checkbox("VQE-Fitness aktivieren", value=True)
    c12, c13, c14, c15 = st.columns(4)
    with c12:
        vqe_weight = st.slider("VQE-Gewicht γ", 0.0, 1.0, 0.35, 0.05)
    with c13:
        vqe_elite_k = st.number_input("Top-K Eliten für VQE", min_value=1, max_value=64, value=8, step=1)
    with c14:
        vqe_qubits = st.number_input("VQE Qubits", min_value=2, max_value=16, value=6, step=1)
    with c15:
        vqe_layers = st.number_input("VQE Layers", min_value=1, max_value=6, value=2, step=1)

    if st.button("🧬 Synthese starten", type="primary", key="run_b"):
        if not objectives_B:
            st.error("Bitte mindestens ein Ziel definieren.")
        else:
            # UI-Startzeit (End-to-End)
            t_ui0 = time.perf_counter()

            with st.spinner("Evolution startet …"):
                try:
                    if use_mycel:
                        mode_label = "Myzel + (optional) VQE"
                        formulas, table, meta = forge.mycelial_quantum_evolution(
                            dll_path=dll_path, gpu_index=int(gpu_index),
                            dataset=dataset_B,
                            objectives=list(objectives_B.keys()),
                            weights=list(objectives_B.values()),
                            population=int(pop_B), steps=int(steps_B),
                            vocab_size=int(vocab_B), max_elements=int(max_elems_B), num_qubits=int(qbits_B),
                            mycel_guidance_strength=float(guidance),
                            mycel_decay=float(decay),
                            mycel_diffusion=float(diffusion),
                            mycel_k_neighbors=int(k_neighbors),
                            mycel_topk_bias=(int(topk_bias) if topk_bias > 0 else None),
                            use_vqe_fitness=bool(use_vqe_fit),
                            vqe_weight=float(vqe_weight),
                            vqe_elite_k=int(vqe_elite_k),
                            vqe_num_qubits=int(vqe_qubits),
                            vqe_layers=int(vqe_layers),
                        )
                    else:
                        mode_label = "Baseline (ohne Myzel)"
                        formulas, table, meta = forge.search_new_material(
                            dll_path=dll_path, gpu_index=int(gpu_index),
                            dataset=dataset_B,
                            objectives=list(objectives_B.keys()),
                            weights=list(objectives_B.values()),
                            population=int(pop_B), steps=int(steps_B),
                            vocab_size=int(vocab_B), max_elements=int(max_elems_B), num_qubits=int(qbits_B),
                        )

                    # UI-Endzeit (End-to-End)
                    t_ui1 = time.perf_counter()
                    t_ui = t_ui1 - t_ui0
                    t_backend = float(meta.get("runtime_s", float("nan")))

                    # Ergebnisse in Session-State
                    st.session_state["df_materials"] = table
                    st.session_state["meta_materials"] = meta
                    st.session_state["last_run_timing"] = {
                        "mode": mode_label,
                        "ui_seconds": t_ui,
                        "backend_seconds": t_backend,
                        "ui_fmt": _fmt_duration(t_ui),
                        "backend_fmt": _fmt_duration(t_backend),
                        "timestamp_end": datetime.now().astimezone().isoformat(timespec="seconds"),
                    }

                    st.success("Synthese abgeschlossen.")

                    # präzise Zeitinfo sofort anzeigen
                    st.info(
                        f"⏱️ **Laufzeiten** — Modus: *{mode_label}*\n\n"
                        f"- End-to-End (UI): **{_fmt_duration(t_ui)}**\n"
                        f"- Backend-Kern (GA/Myzel/VQE): **{_fmt_duration(t_backend)}**\n"
                        f"- Fertig um: {st.session_state['last_run_timing']['timestamp_end']}"
                    )

                except ImportError as e:
                    st.error(f"Fehlendes Paket: {e}. (Tipp: pip install jarvis-tools)")
                except Exception as e:
                    st.exception(e)

    if "df_materials" in st.session_state:
        st.subheader("Vorschläge")
        st.dataframe(st.session_state["df_materials"], width="stretch")
        st.download_button(
            "⤓ CSV exportieren",
            data=st.session_state["df_materials"].to_csv(index=False).encode("utf-8"),
            file_name="material_suggestions.csv",
            mime="text/csv",
        )

        meta = st.session_state.get("meta_materials", {})
        if "pheromone_history" in meta and meta["pheromone_history"]:
            st.subheader("🧪 Myzel – mittlere Pheromonstärke")
            ph = pd.Series(meta["pheromone_history"], name="pher_mean")
            st.line_chart(ph)
            st.caption("Anstieg deutet auf erfolgreiche Lern-/Leiteffekte im Myzel hin.")

        if "vqe_fitness" in meta:
            vf = meta["vqe_fitness"]
            st.info(
                f"VQE-Fitness: {'aktiv' if vf.get('enabled') else 'inaktiv'} • "
                f"γ={vf.get('weight')} • Top-K={vf.get('elite_k')} • "
                f"Q={vf.get('num_qubits')} • L={vf.get('layers')} • Cache={vf.get('cache_size')}"
            )

# ===================================================================
# TAB C – Diagnostik
# ===================================================================
with tabC:
    st.header("C) Diagnostik")

    # Letzte Laufzeit anzeigen
    last = st.session_state.get("last_run_timing")
    if last:
        st.subheader("⏱️ Letzte Laufzeit")
        st.write(
            f"- Modus: **{last.get('mode','')}**  \n"
            f"- End-to-End (UI): **{last.get('ui_fmt','n/a')}**  \n"
            f"- Backend-Kern (GA/Myzel/VQE): **{last.get('backend_fmt','n/a')}**  \n"
            f"- Fertig um: {last.get('timestamp_end','')}"
        )
    else:
        st.info("Noch keine Laufzeit erfasst. Starte eine Synthese in Tab B.")

    st.markdown(
        "Hier siehst du **Trainings- und Laufmetriken** (pro Generation) sowie einen "
        "**Surrogat-Gesundheitscheck** für die aktuell gewählten Materialziele."
    )

    # 1) GA-/Myzel-/VQE-Metriken aus meta['gen_history']
    meta = st.session_state.get("meta_materials", {})
    gen_hist = meta.get("gen_history", None)

    if gen_hist:
        st.subheader("📈 Generationsmetriken")
        gh = pd.DataFrame(gen_hist)
        c1, c2 = st.columns(2)
        with c1:
            if {"gen", "best_norm", "mean_norm"} <= set(gh.columns):
                chart_df = gh.set_index("gen")[["best_norm", "mean_norm"]]
                st.line_chart(chart_df, height=240)
            else:
                st.info("Noch keine Norm-Score-Metriken verfügbar.")

        with c2:
            cols = [c for c in ["pheromone_mean", "vqe_eval", "vqe_cache_size"] if c in gh.columns]
            if cols:
                st.line_chart(gh.set_index("gen")[cols], height=240)
            else:
                st.info("Noch keine Myzel-/VQE-Metriken verfügbar.")

        with st.expander("Rohdaten (gen_history)"):
            st.dataframe(gh, width="stretch", height=320)
            st.download_button(
                "⤓ gen_history.csv",
                data=gh.to_csv(index=False).encode("utf-8"),
                file_name="gen_history.csv",
                mime="text/csv",
            )
    else:
        st.info("Keine Generationsmetriken gefunden. Starte eine Synthese in Tab B.")

    st.markdown("---")

    # 2) Surrogat-Gesundheitscheck
    st.subheader("🩺 Surrogat-Gesundheitscheck (aktueller Datensatz & Ziele)")

    # Aktuelle Einstellungen aus Tab B
    dataset = st.session_state.get("dataset_B", "dft_3d")
    dataset = dataset if isinstance(dataset, str) else "dft_3d"
    objectives_cur = st.session_state.get("obj_builder_B", {})

    if not objectives_cur:
        st.info("Keine Materialziele gewählt (Tab B). Wähle Ziele, dann hier aktualisieren.")
    else:
        with st.spinner("Trainiere Surrogate für Diagnostik …"):
            try:
                # 2.1 Rohdaten laden (unkached, um dict-Spalten-Hashing zu vermeiden)
                df = _load_df(dataset)

                # 2.2 Nur Formeln für Vokabular extrahieren → tuple (hashbar)
                formulas_tuple: Tuple[str, ...] = tuple(df["formula"].astype(str).tolist())

                # 2.3 Vokabular cachen (aus Formeln, hashbar)
                vocab = _cached_vocab_from_formulas(formulas_tuple, max_elems=int(st.session_state.get("vocab_B", 32) or 32))

                # 2.4 Zielspalten sauber abbilden (wie im Backend)
                obj_list = list(objectives_cur.keys())
                df_targets = pd.DataFrame({"formula": df["formula"].astype(str)})
                col_map: Dict[str, str | None] = {}
                for obj in obj_list:
                    lo = obj.lower()
                    if lo == "bandgap":
                        col = "mbj_bandgap" if "mbj_bandgap" in df.columns else "bandgap"
                    elif lo == "formation_energy":
                        col = "formation_energy_peratom" if "formation_energy_peratom" in df.columns else "formation_energy"
                    else:
                        col = obj
                    if col in df.columns:
                        df_targets[obj] = pd.to_numeric(df[col], errors="coerce")
                        col_map[obj] = col
                    else:
                        col_map[obj] = None  # fehlt im Datensatz

                trainable = [o for o in obj_list if col_map.get(o)]
                if not trainable:
                    st.warning("Keine passenden Zielspalten im Datensatz gefunden.")
                else:
                    # 2.5 Surrogate cachen – auf df_targets (schlank & hashbar)
                    sur_map = _cached_surrogates_from_targets(df_targets[["formula"] + trainable], vocab, trainable)

                    # 2.6 Kennzahlen je Ziel
                    rows = []
                    N = len(df_targets)
                    for obj in obj_list:
                        col = col_map.get(obj)
                        if not col:
                            rows.append({
                                "objective": obj,
                                "status": "Spalte fehlt",
                                "valid_count": 0,
                                "nan_rate": 1.0,
                                "||W||₂": np.nan,
                                "bias b": np.nan,
                            })
                            continue
                        y = pd.to_numeric(df_targets[obj], errors="coerce").to_numpy(dtype=np.float64)
                        valid = np.isfinite(y)
                        valid_count = int(np.sum(valid))
                        nan_rate = float(1.0 - valid_count / max(1, N))

                        if obj in sur_map:
                            s = sur_map[obj]
                            w_norm = float(np.linalg.norm(s.W.astype(np.float64)))
                            b = float(s.b)
                            status = "OK"
                        else:
                            w_norm, b, status = np.nan, np.nan, "nicht trainiert"

                        rows.append({
                            "objective": obj,
                            "status": status,
                            "valid_count": valid_count,
                            "nan_rate": round(nan_rate, 4),
                            "||W||₂": round(w_norm, 6) if np.isfinite(w_norm) else np.nan,
                            "bias b": round(b, 6) if np.isfinite(b) else np.nan,
                        })

                    diag_df = pd.DataFrame(rows)
                    st.dataframe(diag_df, width="stretch", height=280)
                    st.download_button(
                        "⤓ surrogate_health.csv",
                        data=diag_df.to_csv(index=False).encode("utf-8"),
                        file_name="surrogate_health.csv",
                        mime="text/csv",
                    )
                    st.caption(
                        "Hinweis: Hohe **nan_rate** bedeutet viele fehlende Werte im Datensatz. "
                        "**||W||₂≈0** deutet auf schwaches/unsicheres Surrogat hin."
                    )
            except Exception as e:
                st.exception(e)
