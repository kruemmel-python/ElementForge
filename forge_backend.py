#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forge_backend.py (Version 5.2 - Chemical Feature Engineering)
============================================================
- Implementiert einen vollständigen K-Means-Algorithmus auf der GPU.
- Nutzt Batched Matrix Multiplication für eine speichereffiziente Berechnung.
- Führt "Constrained Optimization" durch: Training nur auf stabilen Materialien.
- NEU: Implementiert Feature Engineering, um chemische Informationen ('chemsys')
  als One-Hot-Encoding für das Modelltraining nutzbar zu machen.
"""
from __future__ import annotations

import ctypes as C
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from materials_mp_connector import MaterialsProjectConnector, ColumnMap

def autodetect_dll() -> str:
    if sys.platform == "win32":
        lib_name = "CipherCore_OpenCl.dll"
    elif sys.platform == "linux":
        lib_name = "libopencl_driver.so"
    elif sys.platform == "darwin":
        lib_name = "libopencl_driver.dylib"
    else:
        return "CipherCore_OpenCl.dll"

    search_paths = [Path.cwd(), Path(__file__).parent, Path.cwd() / "build", Path.cwd() / "Release", Path.cwd() / "Debug"]
    for path in search_paths:
        candidate = path / lib_name
        if candidate.exists() and candidate.is_file():
            print(f"[Info] Treiber-DLL automatisch erkannt: {candidate}")
            return str(candidate)
    
    print(f"[Warnung] Konnte '{lib_name}' nicht automatisch erkennen. Bitte Pfad manuell angeben.")
    return lib_name

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# ============================================================================
# ===  SCHICHT 1: CIPHERCORE TREIBER-WRAPPER (ERWEITERT)                    ===
# ============================================================================

class CipherCore:
    def __init__(self, dll_path: str | Path, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)
        self.path = str(dll_path)
        try:
            self.lib = C.CDLL(self.path)
        except OSError as e:
            raise OSError(f"Konnte DLL nicht laden: {self.path}") from e

        self._def("initialize_gpu", [C.c_int], C.c_int, required=True)
        self._def("shutdown_gpu", [C.c_int], None)
        self._def("allocate_gpu_memory", [C.c_int, C.c_size_t], C.c_void_p, required=True)
        self._def("free_gpu_memory", [C.c_int, C.c_void_p], None)
        self._def("write_host_to_gpu_blocking", [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p], C.c_int, required=True)
        self._def("read_gpu_to_host_blocking", [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p], C.c_int, required=True)
        self._def("finish_gpu", [C.c_int], C.c_int)

        self.has_matmul = self._def("execute_matmul_batched_on_gpu", [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int)
        self.has_pairwise = self._def("execute_pairwise_similarity_gpu", [C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int], C.c_int)
        self.has_assign = self._def("execute_dynamic_token_assignment_gpu", [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int)
        self.has_segsum = self._def("execute_proto_segmented_sum_gpu", [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int], C.c_int)
        self.has_proupd = self._def("execute_proto_update_step_gpu", [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_float, C.c_int, C.c_int], C.c_int)
        
        self.has_mycel = self._def("subqg_init_mycel", [C.c_int, C.c_int, C.c_int, C.c_int], C.c_int)
        if self.has_mycel:
            self._def("subqg_set_active_T", [C.c_int, C.c_int], C.c_int)
            self._def("set_neighbors_sparse", [C.c_int, np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("step_pheromone_reinforce", [C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("step_pheromone_diffuse_decay", [C.c_int], C.c_int)
            self._def("read_pheromone_slice", [C.c_int, C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("set_pheromone_gains", [C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"), C.c_int], C.c_int)
            self._def("set_diffusion_params", [C.c_int, C.c_float, C.c_float], C.c_int)

        if not self.lib.initialize_gpu(self.gpu_index):
            raise RuntimeError(f"GPU-Init fehlgeschlagen: {dll_path}")

    def _def(self, name: str, argtypes: list, restype, required: bool = False) -> bool:
        if hasattr(self.lib, name):
            fn = getattr(self.lib, name); fn.argtypes, fn.restype = argtypes, restype
            return True
        if required: raise AttributeError(f"Fehlende DLL-Funktion: {name}")
        return False
    def malloc(self, nb: int) -> C.c_void_p:
        p = self.lib.allocate_gpu_memory(self.gpu_index, nb)
        if not p: raise MemoryError(f"GPU-Malloc fehlgeschlagen ({nb} bytes)")
        return p
    def free(self, p: C.c_void_p | None):
        if p: self.lib.free_gpu_memory(self.gpu_index, p)
    def h2d(self, a: np.ndarray, p: C.c_void_p | None = None) -> C.c_void_p:
        ac = np.ascontiguousarray(a, dtype=np.float32); p = p or self.malloc(ac.nbytes)
        if not self.lib.write_host_to_gpu_blocking(self.gpu_index, p, 0, ac.nbytes, ac.ctypes.data_as(C.c_void_p)):
            raise RuntimeError("H2D-Transfer fehlgeschlagen.")
        return p
    def d2h(self, p: C.c_void_p, sh: np.ndarray) -> np.ndarray:
        o = np.empty_like(sh, dtype=np.float32)
        if not self.lib.read_gpu_to_host_blocking(self.gpu_index, p, 0, o.nbytes, o.ctypes.data_as(C.c_void_p)):
            raise RuntimeError("D2H-Transfer fehlgeschlagen.")
        return o

    def pairwise_similarity(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        if not self.has_pairwise: return (X @ X.T).astype(np.float32, copy=False)
        dX, dS = self.h2d(X), self.malloc(N * N * 4)
        try:
            if not self.lib.execute_pairwise_similarity_gpu(self.gpu_index, dX, dS, N, D): raise RuntimeError("pairwise_similarity fehlgeschlagen.")
            return self.d2h(dS, np.empty((N, N), dtype=np.float32))
        finally: self.free(dX); self.free(dS)

    def cross_similarity_batched(self, A: np.ndarray, B: np.ndarray, rows_per_batch: int = 1024) -> np.ndarray:
        A, B = np.ascontiguousarray(A, dtype=np.float32), np.ascontiguousarray(B, dtype=np.float32)
        N, D = A.shape; T, D2 = B.shape
        if D != D2: raise ValueError(f"Dimensions-Mismatch für Cross-Similarity: A.shape[1] ({D}) != B.shape[1] ({D2})")
        if not self.has_matmul or N <= rows_per_batch: return (A @ B.T).astype(np.float32, copy=False)

        nb = (N + rows_per_batch - 1) // rows_per_batch; M = rows_per_batch
        A_pad = np.zeros((nb * M, D), dtype=np.float32); A_pad[:N, :] = A
        A_batched = A_pad.reshape(nb, M, D)
        B_batched_T = np.broadcast_to(B.T, (nb, D, T)).copy()

        dA, dB = self.h2d(A_batched), self.h2d(B_batched_T)
        dC = self.malloc(nb * M * T * 4)
        try:
            if not self.lib.execute_matmul_batched_on_gpu(self.gpu_index, dA, dB, dC, nb, M, T, D): raise RuntimeError("batched matmul fehlgeschlagen.")
            return self.d2h(dC, np.empty((nb, M, T), dtype=np.float32)).reshape(nb * M, T)[:N, :]
        finally: self.free(dA); self.free(dB); self.free(dC)

    def assign_to_prototypes(self, X_gpu, P_gpu, N, T, D) -> C.c_void_p:
        d_assign = self.malloc(N * 4)
        if not self.lib.execute_dynamic_token_assignment_gpu(self.gpu_index, X_gpu, P_gpu, d_assign, 1, N, D, T):
            raise RuntimeError("execute_dynamic_token_assignment_gpu fehlgeschlagen.")
        return d_assign

    def kmeans_update_step_gpu(self, P_gpu, X_gpu, assign_gpu, N, T, D, lr):
        d_sums, d_counts = self.malloc(T * D * 4), self.malloc(T * 4)
        try:
            if not self.lib.execute_proto_segmented_sum_gpu(self.gpu_index, X_gpu, assign_gpu, d_sums, d_counts, N, D, T):
                raise RuntimeError("execute_proto_segmented_sum_gpu fehlgeschlagen.")
            if not self.lib.execute_proto_update_step_gpu(self.gpu_index, P_gpu, d_sums, d_counts, C.c_float(lr), D, T):
                raise RuntimeError("execute_proto_update_step_gpu fehlgeschlagen.")
        finally: self.free(d_sums); self.free(d_counts)
    
    def mycel_init(self, n, k):
        if not self.lib.subqg_init_mycel(self.gpu_index,n,1,k): raise RuntimeError("subqg_init_mycel fehlgeschlagen.")
    def mycel_set_active_T(self, t):
        if not self.lib.subqg_set_active_T(self.gpu_index,t): raise RuntimeError("subqg_set_active_T fehlgeschlagen.")
    def mycel_set_neighbors(self, nidx):
        if not self.lib.set_neighbors_sparse(self.gpu_index,np.ascontiguousarray(nidx,dtype=np.int32)): raise RuntimeError("set_neighbors_sparse fehlgeschlagen.")
    def mycel_set_params(self, diff, dec, gains):
        if not self.lib.set_diffusion_params(self.gpu_index,C.c_float(diff),C.c_float(dec)): raise RuntimeError("set_diffusion_params fehlgeschlagen.")
        if not self.lib.set_pheromone_gains(self.gpu_index,np.ascontiguousarray(gains,dtype=np.float32),gains.size): raise RuntimeError("set_pheromone_gains fehlgeschlagen.")
    def mycel_reinforce(self, d):
        if not self.lib.step_pheromone_reinforce(self.gpu_index,np.ascontiguousarray(d,dtype=np.float32)): raise RuntimeError("step_pheromone_reinforce fehlgeschlagen.")
    def mycel_diffuse_decay(self):
        if not self.lib.step_pheromone_diffuse_decay(self.gpu_index): raise RuntimeError("step_pheromone_diffuse_decay fehlgeschlagen.")
    def mycel_read(self, buf):
        if not self.lib.read_pheromone_slice(self.gpu_index,0,np.ascontiguousarray(buf,dtype=np.float32)): raise RuntimeError("read_pheromone_slice fehlgeschlagen.")
        return buf
        
    def __del__(self):
        if hasattr(self, 'lib') and self.lib: self.lib.shutdown_gpu(self.gpu_index)

# ============================================================================
# ===  DATENLADEN & GPU-KMEANS                                            ===
# ============================================================================
def load_dataframe(source_type, path_or_query, api_key=None, column_map=None):
    if source_type == "Lokale CSV":
        if hasattr(path_or_query, "read"): return pd.read_csv(path_or_query)
        if not Path(str(path_or_query)).exists(): raise FileNotFoundError(f"CSV nicht gefunden: {path_or_query}")
        df = pd.read_csv(path_or_query)
        if df.shape[0] < 2: raise ValueError("Trainingsdaten leer oder enthalten nur einen Punkt.")
        return df
    elif source_type == "Materials Project":
        if not isinstance(path_or_query, dict) or column_map is None: raise ValueError("Für MP wird 'query' und 'column_map' benötigt.")
        df = MaterialsProjectConnector(column_map, api_key, path_or_query).fetch()
        if df.shape[0] < 2: raise ValueError("Materials Project lieferte <2 Zeilen.")
        return df
    raise ValueError(f"Unbekannter Quelltyp: {source_type}")

def gpu_kmeans_full(cc: CipherCore, Xs_gpu: C.c_void_p, N: int, D: int, T: int, iters: int = 8, lr: float = 1.0, seed: int = 42) -> np.ndarray:
    print(f"  - Starte GPU-KMeans: {N} Samples -> {T} Prototypen...")
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(N, size=T, replace=False)
    X_host = cc.d2h(Xs_gpu, np.empty((N, D), dtype=np.float32))
    prototypes_host = X_host[initial_indices].copy()
    P_gpu = cc.h2d(prototypes_host)
    
    assign_gpu = None
    try:
        for i in range(iters):
            assign_gpu = cc.assign_to_prototypes(Xs_gpu, P_gpu, N, T, D)
            cc.kmeans_update_step_gpu(P_gpu, Xs_gpu, assign_gpu, N, T, D, lr=lr)
            cc.free(assign_gpu); assign_gpu = None
        return cc.d2h(P_gpu, np.empty((T, D), dtype=np.float32))
    finally:
        cc.free(P_gpu)
        if assign_gpu: cc.free(assign_gpu)

# ============================================================================
# ===  MPG MODELL & EVOLUTION                                             ===
# ============================================================================
@dataclass(slots=True)
class MPGModel:
    prototypes: np.ndarray
    prototype_properties: np.ndarray
    neighbor_indices: np.ndarray
    scaler: StandardScaler
    # NEU: Speichere die Liste der chemischen Features für die Vorhersage
    all_elements: List[str]
    physical_features: List[str]

# ... (crossover, mutate, etc. bleiben unverändert) ...
def crossover_blend(p1: np.ndarray, p2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return alpha * p1 + (1.0 - alpha) * p2

def mutate_gaussian(ind: np.ndarray, space: Dict[str, Tuple[float, float]], strength: float) -> np.ndarray:
    mutated = ind.copy()
    rng = np.random.default_rng()
    for i, (key, (low, high)) in enumerate(space.items()):
        if key == 'chemsys': continue # chemsys nicht mutieren
        scale = (high - low) * strength
        mutated[i] += rng.normal(0, scale)
    return mutated

def _clip_to_space(ind: np.ndarray, space: Dict[str, Tuple[float, float]]) -> np.ndarray:
    clipped = ind.copy()
    i = 0
    for key, (low, high) in space.items():
        if key == 'chemsys':
            i += 1
            continue
        clipped[i] = np.clip(clipped[i], low, high)
        i += 1
    return clipped

def _apply_guidance(norm_ind: np.ndarray, pheromones: np.ndarray, strength: float) -> np.ndarray:
    if not np.any(pheromones) or pheromones.sum() < 1e-9:
        return norm_ind
    
    guidance_vec = pheromones / pheromones.sum()
    guided = norm_ind * (1.0 - strength) + guidance_vec * strength
    return np.clip(guided, 0, 1)

def _adaptive_softmax_row(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x));
    if std < 1e-9: return np.full(x.shape, 1.0/x.size, dtype=np.float32)
    z = (x - float(np.mean(x))) / std; e = np.exp(z); return e / e.sum()

# NEUE, INTELLIGENTE VERSION von train_mpg_surrogates
def train_mpg_surrogates(cc: CipherCore, df: pd.DataFrame, search_space, obj_list, n_prototypes=64, k_neighbors=8, use_gpu_kmeans=True) -> Dict[str, MPGModel]:
    df_stable = df[df['e_above_hull'] <= 0.02].copy().reset_index(drop=True)
    print(f"[Info] Nach Stabilitätsfilter: {len(df_stable)} von {len(df)} Materialien verbleiben für das Training.")
    
    if len(df_stable) < n_prototypes:
        print(f"[Warnung] Nach Filterung sind zu wenige Datenpunkte ({len(df_stable)}) übrig. Reduziere Prototypen auf {len(df_stable)}.")
        n_prototypes = max(2, len(df_stable))

    print("[Info] Erzeuge chemische Features aus 'chemsys' (One-Hot-Encoding)...")
    all_elements = sorted(list(set(el for chemsys in df_stable['chemsys'] for el in chemsys.split('-'))))
    
    chem_features = []
    for chemsys in df_stable['chemsys']:
        elements_in_row = set(chemsys.split('-'))
        chem_features.append([1.0 if el in elements_in_row else 0.0 for el in all_elements])
    
    chem_features_df = pd.DataFrame(chem_features, columns=[f"has_{el}" for el in all_elements])

    physical_features = [f for f in search_space.keys() if f in df_stable.columns]
    X_physical = df_stable[physical_features].to_numpy(dtype=np.float32)
    X_chem = chem_features_df.to_numpy(dtype=np.float32)
    X_train = np.hstack([X_physical, X_chem])
    
    print(f"[Info] Training mit {X_train.shape[1]} Features insgesamt ({X_physical.shape[1]} physikalische, {X_chem.shape[1]} chemische).")

    n_samples = X_train.shape[0]
    if n_samples < 2: raise ValueError("Nach Stabilitätsfilterung sind zu wenige Trainingsdaten übrig.")
    T = min(max(2, n_prototypes), n_samples)
    k = min(max(0, k_neighbors), T - 1)
    
    scaler = StandardScaler().fit(X_train); Xs = scaler.transform(X_train).astype(np.float32)
    
    Xs_gpu = cc.h2d(Xs)
    try:
        if use_gpu_kmeans and cc.has_assign and cc.has_segsum and cc.has_proupd:
            prototypes_scaled = gpu_kmeans_full(cc, Xs_gpu, *Xs.shape, T)
        else:
            prototypes_scaled = Xs[np.random.default_rng(42).choice(n_samples, size=T, replace=False)]
    finally:
        cc.free(Xs_gpu)

    if k > 0: sim_matrix = cc.pairwise_similarity(prototypes_scaled); np.fill_diagonal(sim_matrix, -np.inf); neighbor_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    else: neighbor_indices = np.zeros((T, 0), dtype=np.int32)

    final_assign = KMeans(n_clusters=T, init=prototypes_scaled, n_init=1).fit(Xs).labels_
    surrogates = {}
    
    obj = 'band_gap'
    if obj not in df_stable.columns:
         raise ValueError("Spalte 'band_gap' nicht in den gefilterten Daten gefunden.")

    y = df_stable[obj].to_numpy()
    valid_y = y[np.isfinite(y)]
    global_mean = np.mean(valid_y) if valid_y.size > 0 else 0.0
    proto_vals = np.array([np.mean(y[final_assign==i][np.isfinite(y[final_assign==i])]) if np.any(final_assign==i) else global_mean for i in range(T)], dtype=np.float32)
    surrogates[obj] = MPGModel(prototypes_scaled, proto_vals, neighbor_indices, scaler, all_elements, physical_features)
    
    return surrogates

def predict_with_mpg(cc: CipherCore, model: MPGModel, X_pred: np.ndarray, sim_steps, diffusion, decay) -> np.ndarray:
    # NEU: Erzeuge chemische Features für die Vorhersagedaten
    # Annahme: X_pred enthält nur die physikalischen Features
    chem_features_pred = np.zeros((X_pred.shape[0], len(model.all_elements)), dtype=np.float32)
    X_pred_full = np.hstack([X_pred, chem_features_pred])

    T, k = model.prototypes.shape[0], model.neighbor_indices.shape[1]
    Xs = model.scaler.transform(X_pred_full)
    sims = cc.cross_similarity_batched(Xs, model.prototypes)
    
    if k == 0 or not cc.has_mycel: return model.prototype_properties[np.argmax(sims, axis=1)]
        
    cc.mycel_init(T, k); cc.mycel_set_active_T(T); cc.mycel_set_neighbors(model.neighbor_indices)
    cc.mycel_set_params(diffusion, decay, np.ones(1, dtype=np.float32))
    
    preds, buf = np.zeros(X_pred.shape[0]), np.zeros(T*k)
    for i in range(X_pred.shape[0]):
        cc.mycel_reinforce(np.zeros(T)); cc.mycel_diffuse_decay(); cc.mycel_diffuse_decay()
        imp = _adaptive_softmax_row(sims[i,:])
        cc.mycel_reinforce(imp); [cc.mycel_diffuse_decay() for _ in range(sim_steps)]
        nodes = cc.mycel_read(buf).reshape((T,k)).sum(axis=1)
        s = nodes.sum()
        preds[i] = np.dot(nodes/s, model.prototype_properties) if s > 1e-9 else model.prototype_properties[np.argmax(imp)]
    return preds

# FINALE, KORREKTE VERSION von score_formulations
def score_formulations(pool: np.ndarray, predictions: Dict, objectives: Dict, search_space: Dict) -> Tuple[np.ndarray, np.ndarray]:
    all_values = predictions.copy()
    search_space_keys = list(search_space.keys())
    for col_name in objectives:
        if col_name in search_space_keys and col_name not in all_values:
            col_index = search_space_keys.index(col_name)
            all_values[col_name] = pool[:, col_index]

    obj_keys = [k for k in objectives if k in all_values]
    if not obj_keys: return np.array([]), np.array([])
    
    weights = np.array([objectives[k] for k in obj_keys], dtype=np.float32)
    P = np.stack([all_values[key] for key in obj_keys], axis=1)
    raw_score = np.sum(P * (weights / (np.sum(np.abs(weights)) + 1e-8)), axis=1)

    valid = np.isfinite(raw_score)
    if not np.any(valid): return np.zeros_like(raw_score), raw_score
    
    lo, hi = np.min(raw_score[valid]), np.max(raw_score[valid])
    
    if (hi - lo) < 1e-9:
        return np.ones_like(raw_score) * 0.5, raw_score

    norm_score = (raw_score - lo) / (hi - lo)
    norm_score[~valid] = 0.0
    
    return np.clip(norm_score, 0, 1).astype(np.float32), raw_score.astype(np.float32)

def mycelial_guided_evolution(
    *,
    dll_path: str, gpu_index: int, source_type: str, data_source_arg: str | Dict,
    mp_api_key: str | None, mp_column_map: ColumnMap | None,
    search_space: Dict[str, Tuple[float, float]], objectives: Dict[str, float],
    population: int, steps: int, n_prototypes: int = 64,
    mycel_guidance_strength: float,
    mycel_decay: float, mycel_diffusion: float, mycel_k_neighbors: int,
    **kwargs
) -> tuple[pd.DataFrame, dict]:
    
    t0 = time.perf_counter()
    
    df_train = load_dataframe(source_type, data_source_arg, mp_api_key, mp_column_map)
    
    cc = CipherCore(dll_path, gpu_index=gpu_index)
    try:
        mpg_surrogates = train_mpg_surrogates(cc, df_train, search_space, list(objectives.keys()), n_prototypes=n_prototypes, k_neighbors=mycel_k_neighbors)
        
        D = len(search_space)
        rng = np.random.default_rng(42)
        mins = np.array([v[0] for v in search_space.values()], dtype=np.float32)
        maxs = np.array([v[1] for v in search_space.values()], dtype=np.float32)
        pool = rng.uniform(mins, maxs, size=(population, D)).astype(np.float32)

        gen_history = []
        k_search = min(mycel_k_neighbors, D - 1) if D > 1 else 0
        
        print("\n--- Starte evolutionäre Suche ---")
        for g in range(steps):
            predictions = {obj: predict_with_mpg(cc, model, pool, sim_steps=3, diffusion=mycel_diffusion, decay=mycel_decay) for obj, model in mpg_surrogates.items()}
            norm_score, raw_score = score_formulations(pool, predictions, objectives, search_space)
            
            if cc.has_mycel and k_search > 0:
                cc.mycel_init(D, k_search)
                cc.mycel_set_active_T(D)
                cc.mycel_set_neighbors(np.array([[(i + j + 1) % D for j in range(k_search)] for i in range(D)], dtype=np.int32))
                cc.mycel_set_params(mycel_diffusion, mycel_decay, np.ones(1, dtype=np.float32))

            elite_n = max(2, int(0.2 * population))
            elite_indices = np.argsort(raw_score)[-elite_n:]
            elites = pool[elite_indices].copy()
            
            if cc.has_mycel and k_search > 0:
                reinforce_vec_norm = (np.mean(elites, axis=0) - mins) / (maxs - mins + 1e-9)
                cc.mycel_reinforce(reinforce_vec_norm * mycel_guidance_strength)
                cc.mycel_diffuse_decay()
                pher_nodes = cc.mycel_read(np.zeros(D * k_search, dtype=np.float32)).reshape((D, k_search)).sum(axis=1)
            else:
                pher_nodes = np.zeros(D, dtype=np.float32)

            children = []
            parent_weights = norm_score + 1e-6
            if parent_weights.sum() < 1e-9: parent_weights = None 
            else: parent_weights /= parent_weights.sum()
            
            while len(children) < population - elite_n:
                p1_idx, p2_idx = rng.choice(np.arange(population), 2, replace=False, p=parent_weights)
                child = crossover_blend(pool[p1_idx], pool[p2_idx])
                t = g / max(1, steps-1); mu = (1-t)*0.15 + t*0.03
                child = mutate_gaussian(child, search_space, strength=float(mu))
                child_norm = (child - mins)/(maxs - mins + 1e-9)
                guided_norm = _apply_guidance(child_norm, pher_nodes, mycel_guidance_strength)
                child = guided_norm*(maxs - mins) + mins
                children.append(_clip_to_space(child, search_space))
            
            pool = np.vstack([elites] + children)

            mean_norm_valid = np.mean(norm_score[norm_score > 0]) if np.any(norm_score > 0) else 0.0
            gen_history.append({"gen": g + 1, "best_norm": np.max(norm_score), "mean_norm": mean_norm_valid, "pheromone_mean": pher_nodes.mean()})
            print(f"[Gen {g+1:02d}] Bester Score: {gen_history[-1]['best_norm']:.4f}, Pheromon: {gen_history[-1]['pheromone_mean']:.6f}")
        
        final_predictions = {obj: predict_with_mpg(cc, model, pool, sim_steps=3, diffusion=mycel_diffusion, decay=mycel_decay) for obj, model in mpg_surrogates.items()}
        final_norm, _ = score_formulations(pool, final_predictions, objectives, search_space)
        
        result_df = pd.DataFrame(pool, columns=list(search_space.keys()))
        result_df["score"] = final_norm
        for prop, values in final_predictions.items(): result_df[f'pred_{prop}'] = values
        
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
        meta = { "runtime_s": time.perf_counter() - t0, "gen_history": gen_history, "objectives": objectives }

        return result_df.head(50), meta
    finally:
        del cc