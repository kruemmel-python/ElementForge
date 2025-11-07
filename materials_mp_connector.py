#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
materials_mp_connector.py (v1.3 - E-Above-Hull & Chemsys)
---------------------------------------------------------
- Fordert 'energy_above_hull' und 'chemsys' als Standardfelder an,
  um Stabilitätsfilterung und chemisches Feature Engineering zu ermöglichen.
- Behält die robuste Datenbereinigung und das Fehlerhandling bei.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os
import pandas as pd
import numpy as np

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@dataclass(slots=True)
class ColumnMap:
    features: Dict[str, str]
    targets: Dict[str, str]

def _rename_and_filter(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    required_external_cols = list(cmap.features.values()) + list(cmap.targets.values())
    
    # Handle 'chemsys' which might not be numeric
    is_numeric_col = {col: col != 'chemsys' for col in df.columns}
    
    missing_cols = [col for col in required_external_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Die folgenden Spalten fehlen in der API-Antwort: {missing_cols}")

    # Trenne Features und Targets
    feature_df = df[list(cmap.features.values())].rename(columns={v: k for k, v in cmap.features.items()})
    target_df = df[list(cmap.targets.values())].rename(columns={v: k for k, v in cmap.targets.items()})
    
    out_df = pd.concat([feature_df, target_df], axis=1)
    
    # Konvertiere nur numerische Spalten
    for col in out_df.columns:
        if is_numeric_col.get(cmap.features.get(col, cmap.targets.get(col)), True):
            out_df[col] = pd.to_numeric(out_df[col], errors='coerce')

    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    out_df = out_df.dropna(subset=list(cmap.targets.keys()), how='all').reset_index(drop=True)
    
    for col in cmap.features.keys():
        if out_df[col].isnull().any():
            # Fülle nur numerische Spalten
            if is_numeric_col.get(cmap.features.get(col), True):
                mean_val = out_df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0
                out_df[col] = out_df[col].fillna(mean_val)
            
    return out_df

@dataclass(slots=True)
class MaterialsProjectConnector:
    column_map: ColumnMap
    api_key: str | None = None
    query: Dict[str, Any] | None = None
    fields: List[str] | None = None
    prefer_client: bool = True
    max_n: int = 5000

    def fetch(self) -> pd.DataFrame:
        key = (self.api_key or os.getenv("MP_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("MaterialsProjectConnector: Kein API-Key. Setze MP_API_KEY oder übergebe api_key.")
        
        fields = self._build_fields()
        
        if self.prefer_client and MP_API_AVAILABLE:
            try:
                return self._fetch_via_client(key, fields)
            except Exception as e:
                print(f"[MP Connector] Client-Abfrage fehlgeschlagen, wechsle zum REST-API-Fallback. Fehler: {e!r}")

        if not REQUESTS_AVAILABLE:
             raise ImportError("Das Paket 'requests' ist nicht installiert. Bitte installieren Sie es mit 'pip install requests'.")

        return self._fetch_via_rest(key, fields)

    def _build_fields(self) -> List[str]:
        wanted = set(self.fields or [])
        wanted.update(self.column_map.features.values())
        wanted.update(self.column_map.targets.values())
        # Stelle sicher, dass diese Felder immer angefordert werden
        wanted.update({"material_id", "formula_pretty", "chemsys", "energy_above_hull", "formation_energy_per_atom"})
        return sorted(list(wanted))

    def _fetch_via_client(self, api_key: str, fields: List[str]) -> pd.DataFrame:
        q = self.query or {}
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(fields=fields, limit=self.max_n, **q)
        if not docs:
            raise RuntimeError("MP-Client lieferte 0 Datensätze (prüfe Filter).")
        
        recs = []
        for d in docs:
            rec = {}
            for k in fields:
                # Behandle verschachtelte Felder wie 'spacegroup.symbol'
                try:
                    val = d
                    for part in k.split('.'):
                        val = getattr(val, part)
                    rec[k] = val
                except AttributeError:
                    rec[k] = None
            recs.append(rec)
            
        df = pd.DataFrame(recs)
        return _rename_and_filter(df, self.column_map)

    def _fetch_via_rest(self, api_key: str, fields: List[str]) -> pd.DataFrame:
        base = "https://api.materialsproject.org/materials/summary/"
        headers = {"accept": "application/json", "x-api-key": api_key}
        params = (self.query or {}).copy()
        params["_fields"] = ",".join(fields)

        per_page = 500
        page = 1
        frames: list[pd.DataFrame] = []

        while True:
            params.update({"_page": page, "_limit": per_page})
            r = requests.get(base, headers=headers, params=params, timeout=60)
            try:
                r.raise_for_status()
            except requests.HTTPError as http_err:
                detail = r.text[:300].replace("\n", " ")
                raise requests.HTTPError(
                    f"MP-REST: {r.status_code} {r.reason}. "
                    f"Ursachen: Falscher Parameter oder ungültiger API-Key. "
                    f"URL: {r.url} | Antwort: {detail}"
                ) from http_err

            payload = r.json()
            data = payload.get("data", [])
            if not data: break
            frames.append(pd.DataFrame(data))
            
            if 0 < self.max_n <= sum(len(f) for f in frames): break
            if len(data) < per_page: break
            page += 1

        if not frames:
            raise RuntimeError("MP-REST lieferte 0 Datensätze (prüfe Filter/Query).")
        
        df = pd.concat(frames, ignore_index=True)
        if self.max_n > 0: df = df.head(self.max_n)

        for f in fields:
            if f not in df.columns: df[f] = np.nan

        return _rename_and_filter(df, self.column_map)