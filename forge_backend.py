#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forge_backend.py
================
Vereinheitlichtes Backend für "Forge Studio" mit *echter* Myzel-Integration
und optionaler VQE-Einbindung in die Fitness während der Evolution.

Blöcke:
A) Elemente bewerten (klassisch + Quantum/VQE)
B) Material-Synthese (Evolution + Surrogat)
C) Myzel-Variante (CipherCore-Myzelnetz mit Pheromon-Guidance)
D) Optional: VQE direkt in der Fitness (Top-Eliten, GPU, Caching)

Python 3.12: match/case (PEP 634–636), | (PEP 604), präzisere Fehler (PEP 626).
"""

from __future__ import annotations

import ctypes as C
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from mendeleev import element as m_element
from mendeleev import get_all_elements

# ------------------------------------------------------------
# DLL / CTYPES
# ------------------------------------------------------------

class PauliZTerm(C.Structure):
    _fields_ = [("z_mask", C.c_uint64), ("coefficient", C.c_float)]


class CipherCore:
    """Wrapper um die CipherCore DLL inklusive Myzel- und VQE-Funktionen."""
    def __init__(self, dll_path: str | Path, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)
        self.path = str(dll_path)
        try:
            self.lib = C.CDLL(self.path)
        except OSError as e:
            raise OSError(f"Konnte DLL nicht laden: {self.path}") from e

        # Basis
        self._def("initialize_gpu", [C.c_int], C.c_int, required=True)
        self._def("shutdown_gpu", [C.c_int], None)
        self._def("allocate_gpu_memory", [C.c_int, C.c_size_t], C.c_void_p)
        self._def("free_gpu_memory", [C.c_int, C.c_void_p], None)
        self._def("write_host_to_gpu_blocking",
                  [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p], C.c_int)
        self._def("read_gpu_to_host_blocking",
                  [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p], C.c_int)

        # Kernels
        self.has_matmul = self._def("execute_matmul_on_gpu",
                                    [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p,
                                     C.c_int, C.c_int, C.c_int, C.c_int], C.c_int)
        self.has_similarity = self._def("execute_pairwise_similarity_gpu",
                                        [C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int], C.c_int)
        # VQE
        self.has_vqe = self._def(
            "execute_vqe_gpu",
            [C.c_int, C.c_int, C.c_int,
             np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"), C.c_int,
             C.POINTER(PauliZTerm), C.c_int, C.POINTER(C.c_float), C.c_void_p],
            C.c_int
        )
        # Myzel
        self.has_mycel = self._def("subqg_init_mycel", [C.c_int, C.c_int, C.c_int, C.c_int], C.c_int)
        if self.has_mycel:
            self._def("set_neighbors_sparse",
                      [C.c_int, np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("step_pheromone_reinforce",
                      [C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("step_pheromone_diffuse_decay", [C.c_int], C.c_int)
            self._def("read_pheromone_slice",
                      [C.c_int, C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")], C.c_int)
            self._def("set_pheromone_gains",
                      [C.c_int, np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"), C.c_int], C.c_int)
            self._def("set_diffusion_params", [C.c_int, C.c_float, C.c_float], C.c_int)

        # Init
        ok = self.lib.initialize_gpu(self.gpu_index)
        if not ok:
            raise RuntimeError(f"initialize_gpu({self.gpu_index}) fehlgeschlagen: {self.path}")

    def _def(self, name: str, argtypes: list, restype, required: bool = False) -> bool:
        if hasattr(self.lib, name):
            fn = getattr(self.lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
            return True
        if required:
            raise AttributeError(f"Fehlende DLL-Funktion: {name}")
        return False

    # --- Memory/IO ---
    def malloc(self, nbytes: int) -> C.c_void_p:
        ptr = self.lib.allocate_gpu_memory(self.gpu_index, nbytes)
        if not ptr:
            raise MemoryError(f"GPU-Allocate fehlgeschlagen (bytes={nbytes})")
        return ptr

    def free(self, ptr: C.c_void_p) -> None:
        self.lib.free_gpu_memory(self.gpu_index, ptr)

    def h2d(self, dst: C.c_void_p, arr: np.ndarray) -> None:
        if arr.nbytes == 0:
            return
        ok = self.lib.write_host_to_gpu_blocking(self.gpu_index, dst, 0, arr.nbytes,
                                                 arr.ctypes.data_as(C.c_void_p))
        if not ok:
            raise RuntimeError("Host->GPU Transfer fehlgeschlagen.")

    def d2h(self, src: C.c_void_p, arr: np.ndarray) -> None:
        if arr.nbytes == 0:
            return
        ok = self.lib.read_gpu_to_host_blocking(self.gpu_index, src, 0, arr.nbytes,
                                                arr.ctypes.data_as(C.c_void_p))
        if not ok:
            raise RuntimeError("GPU->Host Transfer fehlgeschlagen.")

    # --- High-Level ---
    def pairwise_similarity(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        if not self.has_similarity:
            return X @ X.T
        N, D = X.shape
        S = np.empty((N, N), dtype=np.float32)
        dX, dS = self.malloc(X.nbytes), self.malloc(S.nbytes)
        try:
            self.h2d(dX, X)
            ok = self.lib.execute_pairwise_similarity_gpu(self.gpu_index, dX, dS, N, D)
            if not ok:
                raise RuntimeError("execute_pairwise_similarity_gpu fehlgeschlagen.")
            self.d2h(dS, S)
        finally:
            self.free(dX); self.free(dS)
        return S

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A.astype(np.float32, copy=False)
        B = B.astype(np.float32, copy=False)
        if not self.has_matmul:
            return A @ B
        M, K = A.shape; K2, N = B.shape
        if K2 != K:
            raise ValueError("Matmul: inkompatible Shapes.")
        C_ = np.empty((M, N), dtype=np.float32)
        dA, dB, dC = self.malloc(A.nbytes), self.malloc(B.nbytes), self.malloc(C_.nbytes)
        try:
            self.h2d(dA, A); self.h2d(dB, B)
            ok = self.lib.execute_matmul_on_gpu(self.gpu_index, dA, dB, dC, 1, M, N, K)
            if not ok:
                raise RuntimeError("execute_matmul_on_gpu fehlgeschlagen.")
            self.d2h(dC, C_)
        finally:
            self.free(dA); self.free(dB); self.free(dC)
        return C_

    def vqe_energy(self, num_qubits: int, layers: int, params: np.ndarray, terms: list[PauliZTerm]) -> float:
        if not self.has_vqe:
            raise NotImplementedError("VQE nicht verfügbar.")
        params = np.ascontiguousarray(params.astype(np.float32))
        out_e = C.c_float(0.0)
        TermArray = PauliZTerm * len(terms)
        h_array = TermArray(*terms)
        ok = self.lib.execute_vqe_gpu(self.gpu_index, int(num_qubits), int(layers),
                                      params, int(params.size), h_array, int(len(terms)),
                                      C.byref(out_e), None)
        if not ok:
            raise RuntimeError("execute_vqe_gpu fehlgeschlagen.")
        return float(out_e.value)

    # --- Myzel: Helfer ---
    def mycel_init(self, nodes: int, k_neighbors: int, layers: int = 1) -> None:
        if not self.has_mycel:
            raise NotImplementedError("Myzel-Funktionalität fehlt in der DLL.")
        ok = self.lib.subqg_init_mycel(self.gpu_index, int(nodes), int(k_neighbors), int(layers))
        if not ok:
            raise RuntimeError("subqg_init_mycel fehlgeschlagen.")

    def mycel_set_neighbors(self, neigh_idx: np.ndarray) -> None:
        neigh_idx = np.ascontiguousarray(neigh_idx.astype(np.int32))
        ok = self.lib.set_neighbors_sparse(self.gpu_index, neigh_idx)
        if not ok:
            raise RuntimeError("set_neighbors_sparse fehlgeschlagen.")

    def mycel_set_params(self, diffusion: float, decay: float, gains: np.ndarray | None = None) -> None:
        ok = self.lib.set_diffusion_params(self.gpu_index, C.c_float(diffusion), C.c_float(decay))
        if not ok:
            raise RuntimeError("set_diffusion_params fehlgeschlagen.")
        if gains is not None:
            gains = np.ascontiguousarray(gains.astype(np.float32))
            ok2 = self.lib.set_pheromone_gains(self.gpu_index, gains, int(gains.size))
            if not ok2:
                raise RuntimeError("set_pheromone_gains fehlgeschlagen.")

    def mycel_reinforce(self, delta: np.ndarray) -> None:
        delta = np.ascontiguousarray(delta.astype(np.float32))
        ok = self.lib.step_pheromone_reinforce(self.gpu_index, delta)
        if not ok:
            raise RuntimeError("step_pheromone_reinforce fehlgeschlagen.")

    def mycel_diffuse_decay(self) -> None:
        ok = self.lib.step_pheromone_diffuse_decay(self.gpu_index)
        if not ok:
            raise RuntimeError("step_pheromone_diffuse_decay fehlgeschlagen.")

    def mycel_read(self, layer: int, out_buf: np.ndarray) -> np.ndarray:
        out = np.ascontiguousarray(out_buf.astype(np.float32))
        ok = self.lib.read_pheromone_slice(self.gpu_index, int(layer), out)
        if not ok:
            raise RuntimeError("read_pheromone_slice fehlgeschlagen.")
        return out

    def __del__(self) -> None:
        try:
            self.lib.shutdown_gpu(self.gpu_index)
        except Exception:
            pass


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalize01(x: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    span = hi - lo
    if span < 1e-9:
        return np.full_like(arr, 0.5, dtype=np.float32)
    out = (arr - lo) / span
    out[~np.isfinite(arr)] = 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def autodetect_dll() -> str:
    for ext in ("dll", "so", "dylib"):
        for name in (f"CipherCore_OpenCl.{ext}", f"libCipherCore_OpenCl.{ext}", f"libCC_OpenCl.{ext}"):
            p = Path(f"./{name}")
            if p.exists() and p.is_file():
                return str(p)
    return "./CipherCore_OpenCl.dll"


# ============================================================================
# A) Elemente bewerten
# ============================================================================

SUPPORTED_ELEMENT_FEATURES = [
    "density", "melting_point", "boiling_point", "atomic_weight",
    "electronegativity_pauling", "vdw_radius", "covalent_radius", "ionization_energy",
]

def _safe_ion_energy(el) -> float | None:
    try:
        d = getattr(el, "ionenergies", None) or getattr(el, "ionization_energies", None)
        return d.get(1) if isinstance(d, dict) else None
    except Exception:
        return None

def load_element_df() -> pd.DataFrame:
    rec = []
    for el in get_all_elements():
        rec.append({
            "atomic_number": el.atomic_number,
            "symbol": el.symbol,
            "name": el.name,
            "atomic_weight": el.atomic_weight,
            "density": el.density,
            "melting_point": el.melting_point,
            "boiling_point": el.boiling_point,
            "electronegativity_pauling": el.electronegativity("pauling"),
            "vdw_radius": el.vdw_radius,
            "covalent_radius": getattr(el, "covalent_radius_pyykko", None),
            "ionization_energy": _safe_ion_energy(el),
        })
    return pd.DataFrame(rec)

def _objective_score(df_feat: pd.DataFrame, objectives: dict[str, float]) -> pd.Series:
    s = np.zeros(len(df_feat), dtype=np.float32)
    for col, w in objectives.items():
        ncol = f"norm_{col}"
        if ncol not in df_feat:
            continue
        v = df_feat[ncol].to_numpy(np.float32)
        s += (1.0 - v) * abs(float(w)) if w < 0 else v * float(w)
    return pd.Series(normalize01(s), index=df_feat.index, name="objective_score")

def _ising_terms_from_objectives(props_norm: dict[str, float], objectives: dict[str, float], num_qubits: int) -> list[PauliZTerm]:
    terms: list[PauliZTerm] = []
    q = 0
    for k, w in objectives.items():
        if k not in props_norm:
            continue
        coeff = -float(w) * (float(props_norm[k]) - 0.5) * 2.0
        t = PauliZTerm()
        t.z_mask = C.c_uint64(1 << (q % max(1, num_qubits)))
        t.coefficient = C.c_float(coeff)
        terms.append(t)
        q += 1
    return terms

def _default_vqe_params(num_qubits: int, layers: int, seed: float = 0.5) -> np.ndarray:
    n = layers * 2 * num_qubits
    return np.sin(0.5 * np.arange(n, dtype=np.float32) + float(seed)).astype(np.float32)

def score_elements(dll_path: str, gpu_index: int,
                   mode: str, objectives: dict[str, float],
                   *, uniqueness_weight: float = 0.3,
                   quantum_weight: float = 0.5,
                   num_qubits: int = 4, ansatz_layers: int = 2) -> pd.DataFrame:
    df = load_element_df()
    feats = [f for f in objectives if f in df.columns]
    if not feats:
        raise ValueError("Keine gültigen Feature-Namen in 'objectives'.")
    df_feat = df[["symbol", "name"] + feats].copy()
    for f in feats:
        df_feat[f"norm_{f}"] = normalize01(df_feat[f])
    df_feat["objective_score"] = _objective_score(df_feat, objectives)

    if mode == "classic":
        X = np.stack([df_feat[f"norm_{f}"].to_numpy(np.float32) for f in feats], axis=1).astype(np.float32)
        try:
            cc = CipherCore(dll_path, gpu_index)
            S = cc.pairwise_similarity(X)
        except Exception as e:
            print(f"[WARNUNG] Fallback CPU-Similarity ({e})", file=sys.stderr)
            S = X @ X.T
        np.fill_diagonal(S, 0.0)
        df_feat["uniqueness_score"] = normalize01(1.0 - np.max(S, axis=1))
        α = float(np.clip(uniqueness_weight, 0.0, 1.0))
        df_feat["final_score"] = normalize01(
            (1.0 - α) * df_feat["objective_score"].to_numpy(np.float32)
            + α * df_feat["uniqueness_score"].to_numpy(np.float32)
        )
        cols = ["symbol", "name", "final_score", "objective_score", "uniqueness_score"] + feats
        return df_feat.sort_values("final_score", ascending=False).reset_index(drop=True)[cols]

    elif mode == "quantum":
        cc = CipherCore(dll_path, gpu_index)
        if not cc.has_vqe:
            raise NotImplementedError("DLL bietet keine VQE-Funktion.")
        energies = []
        for _, row in df_feat.iterrows():
            props = {f: float(row[f"norm_{f}"]) for f in feats}
            terms = _ising_terms_from_objectives(props, objectives, num_qubits)
            if not terms:
                energies.append(np.nan); continue
            params = _default_vqe_params(num_qubits, ansatz_layers, seed=props.get("density", 0.5))
            try:
                e = cc.vqe_energy(num_qubits, ansatz_layers, params, terms)
            except Exception as e0:
                print(f"[VQE] {row['symbol']}: {e0}", file=sys.stderr)
                e = np.nan
            energies.append(e)
        df_feat["vqe_energy"] = energies
        df_feat["quantum_score"] = 1.0 - normalize01(df_feat["vqe_energy"].to_numpy(np.float32))
        β = float(np.clip(quantum_weight, 0.0, 1.0))
        df_feat["final_score"] = normalize01(
            (1.0 - β) * df_feat["objective_score"].to_numpy(np.float32)
            + β * df_feat["quantum_score"].to_numpy(np.float32)
        )
        cols = ["symbol", "name", "final_score", "objective_score", "quantum_score", "vqe_energy"] + feats
        return df_feat.sort_values("final_score", ascending=False).reset_index(drop=True)[cols]
    else:
        raise ValueError(f"Unbekannter Modus: {mode!r}")


# ============================================================================
# B) Material-Synthese (Basis)
# ============================================================================

JARVIS_DATASET_ALIASES: dict[str, str] = {
    "jdft_3d": "dft_3d",
    "jdft_2d": "dft_2d",
    "dft_3d_2021": "dft_3d",
    "dft_2d_2021": "dft_2d",
    "cfid": "cfid_3d",
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")

def load_jarvis_dataframe(dataset: str = "dft_3d") -> pd.DataFrame:
    from jarvis.db.figshare import data as jarvis_data
    key = JARVIS_DATASET_ALIASES.get(dataset, dataset)
    raw = jarvis_data(dataset=key)
    df = pd.DataFrame(raw)
    if "formula" not in df.columns:
        for alt in ("formula_pretty", "compound"):
            if alt in df.columns:
                df["formula"] = df[alt]; break
        else:
            raise ValueError("Keine Formelspalte in JARVIS-Daten gefunden.")
    df = df.dropna(subset=["formula"]).reset_index(drop=True)
    return df

def parse_formula(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for el, nstr in _TOKEN.findall(str(formula)):
        n = float(nstr) if nstr else 1.0
        counts[el] = counts.get(el, 0.0) + n
    if not counts:
        raise ValueError(f"Formel nicht parsebar: {formula!r}")
    return counts

def build_vocab(df: pd.DataFrame, max_elems: int = 32) -> list[str]:
    from collections import Counter
    cnt: Counter[str] = Counter()
    for f in df["formula"].astype(str):
        for el, n in parse_formula(f).items():
            cnt[el] += n
    vocab = [e for e, _ in cnt.most_common(max_elems)]
    vocab.sort(key=lambda x: (len(x), x))
    return vocab

def fraction_features(formula: str, vocab: Sequence[str]) -> np.ndarray:
    comp = parse_formula(formula)
    s = sum(comp.values())
    vec = np.zeros((len(vocab),), dtype=np.float32)
    if s <= 0:
        return vec
    for i, el in enumerate(vocab):
        if el in comp:
            vec[i] = float(comp[el] / s)
    return vec

@dataclass(slots=True)
class Surrogate:
    mu: np.ndarray
    sigma: np.ndarray
    W: np.ndarray
    b: float

def train_surrogate_scalar(Xfrac: np.ndarray, y: np.ndarray) -> Surrogate:
    X = Xfrac.astype(np.float64, copy=False)
    y = y.reshape(-1, 1).astype(np.float64, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    X1 = np.c_[Xn, np.ones((Xn.shape[0], 1))]
    Q, R = np.linalg.qr(X1)
    beta = np.linalg.solve(R, Q.T @ y)
    W = beta[:-1].astype(np.float32)
    b = float(beta[-1, 0])
    return Surrogate(mu.astype(np.float32), sigma.astype(np.float32), W, b)

def gpu_matmul(cc: CipherCore, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if cc.has_matmul:
        return cc.matmul(A, B)
    return A.astype(np.float32) @ B.astype(np.float32)

def predict_scalar_gpu(cc: CipherCore, sur: Surrogate, Xfrac: np.ndarray) -> np.ndarray:
    X = Xfrac.astype(np.float32, copy=False)
    Xn = (X - sur.mu) / sur.sigma
    out = gpu_matmul(cc, Xn, sur.W) + sur.b
    return out.ravel().astype(np.float32)

def _renorm_nonneg(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    s = float(v.sum())
    if s <= 0.0:
        j = int(np.argmax(v))
        out = np.zeros_like(v, dtype=np.float32); out[j] = 1.0
        return out
    return (v / s).astype(np.float32)

def mutate_vec(v: np.ndarray, *, sigma: float = 0.05, strategy: str = "gaussian") -> np.ndarray:
    v = v.astype(np.float32, copy=True)
    match strategy:
        case "gaussian":
            v += np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
        case "cauchy":
            v += (np.random.standard_cauchy(size=v.shape).astype(np.float32) * (sigma * 0.2))
        case "dirichlet":
            alpha = np.full(v.size, 1.5, dtype=np.float64)
            v = np.random.dirichlet(alpha).astype(np.float32)
        case _:
            v += np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
    return _renorm_nonneg(v)

def crossover_sbx(a: np.ndarray, b: np.ndarray, n: float = 2.0) -> np.ndarray:
    a = a.astype(np.float32, copy=False); b = b.astype(np.float32, copy=False)
    u = np.random.rand(*a.shape).astype(np.float32)
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (n + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (n + 1.0))).astype(np.float32)
    child = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    return _renorm_nonneg(child.astype(np.float32))

def _atomic_number(sym: str) -> int:
    try:
        return int(m_element(sym).atomic_number)
    except Exception:
        return 999

def _build_knn_neighbors(vocab: list[str], k: int) -> np.ndarray:
    """
    Baue k-NN über Ordnungszahlen (stetig & reproduzierbar).
    Ergebnisform: (D,k) int32, jeder Knoten hat k Nachbarn.
    """
    D = len(vocab)
    Z = np.array([_atomic_number(s) for s in vocab], dtype=np.int32)
    neigh = np.zeros((D, max(1, k)), dtype=np.int32)
    for i in range(D):
        dz = np.abs(Z - Z[i])
        dz[i] = 10_000
        idx = np.argsort(dz)[:k]
        neigh[i, :len(idx)] = idx.astype(np.int32)
        if len(idx) < k:
            extra = []
            t = 0
            while len(idx) + len(extra) < k:
                extra.append((i + 1 + t) % D)
                t += 1
            neigh[i, :] = np.concatenate([idx, np.array(extra[:k - len(idx)], dtype=np.int32)])
    return neigh.astype(np.int32)

# ---- VQE-Fitness-Helfer -----------------------------------------------------

def _props_norm_from_pool_preds(pred_matrix: np.ndarray, obj_list: Sequence[str]) -> list[dict[str, float]]:
    """
    pred_matrix: shape (N, M) – Spalten entsprechen obj_list.
    Rückgabe: N Dictionaries mit normierten Werten je Ziel.
    """
    props_norm_list: list[dict[str, float]] = []
    for j in range(pred_matrix.shape[1]):
        col = normalize01(pred_matrix[:, j])
        pred_matrix[:, j] = col  # in-place normalisiert
    for i in range(pred_matrix.shape[0]):
        props_norm_list.append({obj_list[j]: float(pred_matrix[i, j]) for j in range(pred_matrix.shape[1])})
    return props_norm_list

def _make_terms_from_props(props_norm: dict[str, float], objectives: dict[str, float], num_qubits: int) -> list[PauliZTerm]:
    return _ising_terms_from_objectives(props_norm, objectives, num_qubits)

def _vec_hash(v: np.ndarray, ndigits: int = 4) -> tuple:
    """
    Stabiler Cache-Key für Kompositionsvektoren.
    Wichtig: NumPy verwendet 'decimals' statt 'ndigits'.
    """
    v32 = v.astype(np.float32, copy=False)
    return tuple(np.round(v32, decimals=int(ndigits)))

# ---- GA Scoring -------------------------------------------------------------

def score_candidates(cc: CipherCore, sur_map: dict[str, Surrogate], X: np.ndarray,
                     objectives: Sequence[str], weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rückgabe:
      norm_score: 0..1 aus gewichteten Surrogaten (für Selektion)
      raw_score:  rohe gewichtete Surrogat-Werte (Monitoring)
      pred_matrix: NxM vor Normierung (einzelne Ziel-Predictions)
    """
    preds = []
    w = np.asarray(weights if weights else np.ones(len(objectives), dtype=np.float32), dtype=np.float32)
    if w.size != len(objectives):
        w = np.ones((len(objectives),), dtype=np.float32)
    w = w / (float(w.sum()) + 1e-8)
    for obj in objectives:
        if obj not in sur_map:
            raise KeyError(f"Kein Surrogat für '{obj}'.")
        preds.append(predict_scalar_gpu(cc, sur_map[obj], X))
    P = np.stack(preds, axis=1).astype(np.float32)  # NxM
    raw = (P @ w).astype(np.float32)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    norm = (raw - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(raw, dtype=np.float32)
    return norm, raw, P

# ---- Basis-Evolution --------------------------------------------------------

def search_new_material(
    *,
    dll_path: str,
    gpu_index: int,
    dataset: str = "dft_3d",
    objectives: Sequence[str] = ("bandgap",),
    weights: Sequence[float] | None = None,
    population: int = 128,
    steps: int = 60,
    vocab_size: int = 32,
    max_elements: int = 4,
    num_qubits: int = 4,
    cc: CipherCore | None = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    Baseline-GA ohne Myzel-Guidance.
    PATCH: Schreibt pro Generation ein Log in meta['gen_history'].
    """
    t0 = time.perf_counter()
    df = load_jarvis_dataframe(dataset)
    vocab = build_vocab(df, max_elems=vocab_size)
    X0 = np.vstack([fraction_features(f, vocab) for f in df["formula"]]).astype(np.float32)
    if X0.shape[0] < 100:
        raise ValueError(f"Zu wenige valide Einträge (N={X0.shape[0]}).")

    obj_list = list(objectives)
    col_map: dict[str, str] = {}
    for obj in obj_list:
        lo = obj.lower()
        if lo == "bandgap":
            col_map[obj] = "mbj_bandgap" if "mbj_bandgap" in df.columns else "bandgap"
        elif lo == "formation_energy":
            col_map[obj] = "formation_energy_peratom" if "formation_energy_peratom" in df.columns else "formation_energy"
        else:
            col_map[obj] = obj
        if col_map[obj] not in df.columns:
            raise ValueError(f"Zielspalte '{obj}' nicht gefunden (versuchte '{col_map[obj]}').")

    ycols: dict[str, np.ndarray] = {obj: pd.to_numeric(df[col_map[obj]], errors="coerce").to_numpy(np.float32)
                                    for obj in obj_list}
    sur_map: dict[str, Surrogate] = {obj: train_surrogate_scalar(X0, ycols[obj]) for obj in obj_list}

    own_cc = False
    if cc is None:
        cc = CipherCore(dll_path, gpu_index=gpu_index)
        own_cc = True

    rng = np.random.default_rng(42)
    D = len(vocab)
    pool = rng.dirichlet(np.ones(D, dtype=np.float32), size=population).astype(np.float32)

    def _limit_k(vec: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or np.count_nonzero(vec) <= k:
            return vec
        idx = np.argsort(vec)[-k:]
        out = np.zeros_like(vec, dtype=np.float32); out[idx] = vec[idx]
        return _renorm_nonneg(out)

    gen_history: list[dict] = []
    global_best_norm = -np.inf
    global_best_vec: np.ndarray | None = None

    for g in range(steps):
        norm_score, raw_score, _ = score_candidates(cc, sur_map, pool, obj_list, list(weights or []))

        # --- PATCH: Generation-Metriken loggen
        gen_entry = {
            "gen": int(g + 1),
            "best_raw": float(np.max(raw_score)),
            "mean_raw": float(np.mean(raw_score)),
            "std_raw": float(np.std(raw_score)),
            "best_norm": float(np.max(norm_score)),
            "mean_norm": float(np.mean(norm_score)),
            "std_norm": float(np.std(norm_score)),
            # baseline: kein Myzel/VQE
            "pheromone_mean": None,
            "vqe_eval": 0,
            "vqe_cache_size": 0,
        }
        gen_history.append(gen_entry)

        if gen_entry["best_norm"] > global_best_norm:
            global_best_norm = gen_entry["best_norm"]
            global_best_vec = pool[int(np.argmax(norm_score))].copy()

        elite_n = max(1, int(round(0.25 * population)))
        elite_idx = np.argsort(norm_score)[-elite_n:]
        elites = pool[elite_idx].copy()
        parents = pool[np.argsort(norm_score)][-max(elite_n * 4, population):]

        children: list[np.ndarray] = []
        while len(children) < (population - elite_n):
            i, j = rng.integers(0, len(parents), size=2)
            child = crossover_sbx(parents[i], parents[j], n=2.0)
            child = mutate_vec(child, sigma=0.05, strategy="gaussian")
            child = _limit_k(child, max_elements)
            children.append(child)
        pool = np.vstack([elites] + children)

        print(f"[Gen {g + 1:02d}] best_norm={gen_entry['best_norm']:.6f}  mean_norm={gen_entry['mean_norm']:.6f}")

    # Kandidaten → Formeln
    final_norm, final_raw, _ = score_candidates(cc, sur_map, pool, obj_list, list(weights or []))
    top_idx = np.argsort(final_norm)[-20:]
    candidates: list[np.ndarray] = []
    if global_best_vec is not None:
        candidates.append(global_best_vec)
    for i in top_idx:
        x = pool[i]
        if not any(np.allclose(x, y, atol=1e-4) for y in candidates):
            candidates.append(x)
    if not candidates:
        candidates.append(pool[int(np.argmax(final_norm))])

    def vec_to_formula(v: np.ndarray, thr: float = 0.05) -> str:
        parts = [(vocab[i], float(v[i])) for i in range(len(vocab)) if v[i] >= thr]
        if not parts:
            j = int(np.argmax(v))
            parts = [(vocab[j], float(v[j]))]
        fracs = np.array([p[1] for p in parts], dtype=np.float64)
        fracs = fracs / float(np.min(fracs))
        nums = np.rint(fracs / 0.25).astype(int)
        nums[nums < 1] = 1
        return "".join([f"{el}{'' if n == 1 else n}" for (el, _), n in zip(parts, nums)])

    formulas = [vec_to_formula(v) for v in candidates]
    Xcand = np.vstack(candidates).astype(np.float32)

    pred_cols: dict[str, np.ndarray] = {}
    for obj in obj_list:
        pred_cols[f"pred_{obj}"] = predict_scalar_gpu(cc, sur_map[obj], Xcand)

    w = np.asarray(weights if weights else np.ones(len(obj_list), dtype=np.float32), dtype=np.float32)
    w = (w / (float(w.sum()) + 1e-8)).astype(np.float32)
    P = np.stack([pred_cols[f"pred_{o}"] for o in obj_list], axis=1).astype(np.float32)
    final_raw_score = (P @ w).astype(np.float32)

    table = pd.DataFrame({
        "formula_suggested": formulas,
        "score": final_raw_score,
        **{k: v for k, v in pred_cols.items()},
    }).sort_values("score", ascending=False).reset_index(drop=True)

    meta = {
        "objectives": obj_list,
        "weights": w.tolist(),
        "runtime_s": time.perf_counter() - t0,
        "gpu": f"index {gpu_index}",
        "population": int(population),
        "steps": int(steps),
        "vocab_size": int(vocab_size),
        "max_elements": int(max_elements),
        "gen_history": gen_history,               # <-- PATCH: Historie verfügbar
        "pheromone_history": [],                  # baseline: leer
        "mycel_params": None,
        "vqe_fitness": {"enabled": False},
    }
    if own_cc:
        del cc
    return formulas, table, meta


# ============================================================================
# C) Myzel-Variante (mit optionaler VQE-Fitness)
# ============================================================================

def mycelial_quantum_evolution(
    *,
    dll_path: str,
    gpu_index: int,
    dataset: str = "dft_3d",
    objectives: Sequence[str] = ("bandgap",),
    weights: Sequence[float] | None = None,
    population: int = 128,
    steps: int = 60,
    vocab_size: int = 32,
    max_elements: int = 4,
    num_qubits: int = 4,
    # Myzel
    mycel_guidance_strength: float = 0.3,
    mycel_decay: float = 0.05,
    mycel_diffusion: float = 0.02,
    mycel_k_neighbors: int = 4,
    mycel_topk_bias: int | None = None,
    # VQE-Fitness
    use_vqe_fitness: bool = True,
    vqe_weight: float = 0.35,
    vqe_elite_k: int = 8,
    vqe_num_qubits: int = 6,
    vqe_layers: int = 2,
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    GA mit Myzel-Guidance + optionaler VQE-Fitness.
    PATCH: Schreibt pro Generation ein Log in meta['gen_history'] inkl. pheromone_mean & vqe_eval.
    """
    t0 = time.perf_counter()
    # 1) Daten & Vokabular
    df = load_jarvis_dataframe(dataset)
    vocab = build_vocab(df, max_elems=vocab_size)
    D = len(vocab)

    # 2) Surrogate
    obj_list = list(objectives)
    col_map: dict[str, str] = {}
    for obj in obj_list:
        lo = obj.lower()
        if lo == "bandgap":
            col_map[obj] = "mbj_bandgap" if "mbj_bandgap" in df.columns else "bandgap"
        elif lo == "formation_energy":
            col_map[obj] = "formation_energy_peratom" if "formation_energy_peratom" in df.columns else "formation_energy"
        else:
            col_map[obj] = obj
        if col_map[obj] not in df.columns:
            raise ValueError(f"Zielspalte '{obj}' nicht gefunden (versuchte '{col_map[obj]}').")
    X0 = np.vstack([fraction_features(f, vocab) for f in df["formula"]]).astype(np.float32)
    ycols: dict[str, np.ndarray] = {obj: pd.to_numeric(df[col_map[obj]], errors="coerce").to_numpy(np.float32)
                                    for obj in obj_list}
    sur_map: dict[str, Surrogate] = {obj: train_surrogate_scalar(X0, ycols[obj]) for obj in obj_list}

    # 3) CipherCore + Myzel
    cc = CipherCore(dll_path, gpu_index=gpu_index)
    if not cc.has_mycel:
        raise NotImplementedError("DLL enthält keine Myzel-Funktionen (subqg_*).")

    k = int(max(1, mycel_k_neighbors))
    cc.mycel_init(nodes=D, k_neighbors=k, layers=1)
    neigh_idx = _build_knn_neighbors(vocab, k=k)
    cc.mycel_set_neighbors(neigh_idx)

    gains = np.array([float(np.clip(mycel_guidance_strength, 0.0, 1.0))], dtype=np.float32)
    cc.mycel_set_params(diffusion=float(mycel_diffusion), decay=float(mycel_decay), gains=gains)

    # 4) GA-Setup
    rng = np.random.default_rng(42)
    population = int(population)
    steps = int(steps)
    max_elements = int(max_elements)
    pool = rng.dirichlet(np.ones(D, dtype=np.float32), size=population).astype(np.float32)

    def _limit_k(vec: np.ndarray, kmax: int) -> np.ndarray:
        if kmax <= 0 or np.count_nonzero(vec) <= kmax:
            return vec
        idx = np.argsort(vec)[-kmax:]
        out = np.zeros_like(vec, dtype=np.float32); out[idx] = vec[idx]
        return _renorm_nonneg(out)

    gen_history: list[dict] = []
    pher_history_mean: list[float] = []
    pher_buf = np.zeros((D,), dtype=np.float32)

    global_best_comb = -np.inf
    global_best_vec: np.ndarray | None = None

    # VQE-Cache: Hash(vec) -> Energie
    vqe_cache: dict[tuple, float] = {}

    # 5) Schleife
    for g in range(steps):
        # Surrogat-Bewertung
        norm_score, raw_score, P_pred = score_candidates(cc, sur_map, pool, obj_list, list(weights or []))

        # Pheromon lesen (für Guidance)
        pher = cc.mycel_read(layer=0, out_buf=pher_buf)
        pher_mean = float(np.mean(pher))
        pher_history_mean.append(pher_mean)

        # Eliten/Eltern
        elite_n = max(1, int(round(0.25 * population)))
        elite_idx = np.argsort(norm_score)[-elite_n:]
        elites = pool[elite_idx].copy()
        parents = pool[np.argsort(norm_score)][-max(elite_n * 4, population):]

        vqe_eval_count = 0

        # ---------------- VQE in Fitness (optional) ----------------
        if use_vqe_fitness and cc.has_vqe and vqe_weight > 0.0:
            # Normierte objektweise Surrogat-Preds → props_norm je Kandidat
            props_norm_list = _props_norm_from_pool_preds(P_pred.copy(), obj_list)  # NxM -> [{'obj': norm_val}, ...]

            # Wähle Top-K Eliten (nach Surrogat) für echte VQE
            k_eval = int(max(1, min(vqe_elite_k, elites.shape[0])))
            elite_eval_idx = elite_idx[-k_eval:]  # absolute Indizes im Pool

            # VQE-Energien sammeln
            energies = np.full(pool.shape[0], np.nan, dtype=np.float32)

            for idx in elite_eval_idx:
                vec = pool[idx]
                hkey = _vec_hash(vec, ndigits=4)  # Achtung: in _vec_hash wurde 'decimals' gefixt
                if hkey in vqe_cache:
                    energies[idx] = vqe_cache[hkey]
                    continue

                # Hamilton-Terms aus normierten Surrogaten dieses Kandidaten
                props_norm = props_norm_list[idx]
                # Wähle Gewichte konsistent zur Gesamtfitness
                w_for_terms = {k: float(v) for k, v in zip(obj_list, (weights or [1.0] * len(obj_list)))}
                terms = _make_terms_from_props(props_norm, w_for_terms, vqe_num_qubits)
                if not terms:
                    continue
                # Parameter: deterministische Seed-Wahl aus Vektor
                seed = float(np.dot(vec, np.arange(vec.size, dtype=np.float32)) % 1.0)
                params = _default_vqe_params(vqe_num_qubits, vqe_layers, seed=seed)
                try:
                    e = cc.vqe_energy(vqe_num_qubits, vqe_layers, params, terms)
                    energies[idx] = e
                    vqe_cache[hkey] = float(e)
                    vqe_eval_count += 1
                except Exception as e0:
                    print(f"[VQE-Fitness] Gen{g+1} idx={idx}: {e0}", file=sys.stderr)

            # Normiere nur die berechneten Energien
            eval_mask = np.isfinite(energies)
            if np.any(eval_mask):
                e_eval = energies[eval_mask]
                e_norm = (e_eval - float(np.min(e_eval))) / (float(np.max(e_eval) - np.min(e_eval)) + 1e-8)
                # VQE-Score = 1 - normierte Energie
                vqe_score = np.zeros_like(energies, dtype=np.float32)
                vqe_score[eval_mask] = 1.0 - e_norm.astype(np.float32)

                # Kombinierte Fitness: nur dort mischen, wo VQE vorhanden ist
                comb = norm_score.copy()
                comb[eval_mask] = (1.0 - float(vqe_weight)) * norm_score[eval_mask] + float(vqe_weight) * vqe_score[eval_mask]
                norm_score = comb  # für Selektion/Eliten

        # Update global best (nach ggf. kombinierter Fitness)
        best_comb = float(np.max(norm_score))
        if best_comb > global_best_comb:
            global_best_comb = best_comb
            global_best_vec = pool[int(np.argmax(norm_score))].copy()

        # --- PATCH: Generation-Metriken loggen
        gen_entry = {
            "gen": int(g + 1),
            "best_raw": float(np.max(raw_score)),
            "mean_raw": float(np.mean(raw_score)),
            "std_raw": float(np.std(raw_score)),
            "best_norm": float(np.max(norm_score)),
            "mean_norm": float(np.mean(norm_score)),
            "std_norm": float(np.std(norm_score)),
            "pheromone_mean": pher_mean,
            "vqe_eval": int(vqe_eval_count),
            "vqe_cache_size": int(len(vqe_cache)),
        }
        gen_history.append(gen_entry)

        # Verstärkung aus Eliten (verwende *norm_score* für Stabilität)
        elite_idx = np.argsort(norm_score)[-elite_n:]
        elites = pool[elite_idx].copy()
        elite_scores = norm_score[elite_idx]
        es = elite_scores - float(np.min(elite_scores))
        es = es / (float(np.sum(es)) + 1e-8)
        reinforce = np.average(elites, axis=0, weights=es if np.isfinite(es).all() else None).astype(np.float32)
        cc.mycel_reinforce(reinforce)
        cc.mycel_diffuse_decay()

        # Kinder mit Pheromon-Guidance
        children: list[np.ndarray] = []
        while len(children) < (population - elite_n):
            i, j = np.random.randint(0, elites.shape[0], size=2)
            pa = elites[i]; pb = elites[j]
            child = crossover_sbx(pa, pb, n=2.0)
            child = mutate_vec(child, sigma=0.05, strategy="gaussian")
            child = _apply_pheromone_guidance(
                child, pher, strength=float(mycel_guidance_strength),
                topk=(int(mycel_topk_bias) if mycel_topk_bias else None)
            )
            child = _limit_k(child, max_elements)
            children.append(child)
        pool = np.vstack([elites] + children)

        print(f"[Gen {g + 1:02d}] best={best_comb:.3f}  mean={float(np.mean(norm_score)):.3f}  pher_mean={pher_mean:.6f}  vqe_eval={vqe_eval_count}")

    # Abschluss – Kandidaten & Tabelle
    final_norm, _, _ = score_candidates(cc, sur_map, pool, obj_list, list(weights or []))
    top_idx = np.argsort(final_norm)[-20:]
    candidates: list[np.ndarray] = []
    if global_best_vec is not None:
        candidates.append(global_best_vec)
    for i in top_idx:
        x = pool[i]
        if not any(np.allclose(x, y, atol=1e-4) for y in candidates):
            candidates.append(x)
    if not candidates:
        candidates.append(pool[int(np.argmax(final_norm))])

    def vec_to_formula(v: np.ndarray, thr: float = 0.05) -> str:
        parts = [(vocab[i], float(v[i])) for i in range(len(vocab)) if v[i] >= thr]
        if not parts:
            j = int(np.argmax(v)); parts = [(vocab[j], float(v[j]))]
        fracs = np.array([p[1] for p in parts], dtype=np.float64)
        fracs = fracs / float(np.min(fracs))
        nums = np.rint(fracs / 0.25).astype(int); nums[nums < 1] = 1
        return "".join([f"{el}{'' if n == 1 else n}" for (el, _), n in zip(parts, nums)])

    formulas = [vec_to_formula(v) for v in candidates]
    Xcand = np.vstack(candidates).astype(np.float32)

    pred_cols: dict[str, np.ndarray] = {}
    for obj in obj_list:
        pred_cols[f"pred_{obj}"] = predict_scalar_gpu(cc, sur_map[obj], Xcand)

    table = pd.DataFrame({
        "formula_suggested": formulas,
        "score": final_norm[np.argsort(final_norm)[-len(formulas):][::-1]],
        **{k: v for k, v in pred_cols.items()},
    }).reset_index(drop=True)

    meta = {
        "objectives": obj_list,
        "weights": list(weights or [1.0] * len(obj_list)),
        "runtime_s": time.perf_counter() - t0,
        "gpu": f"index {gpu_index}",
        "population": int(population),
        "steps": int(steps),
        "vocab_size": int(vocab_size),
        "max_elements": int(max_elements),
        "gen_history": gen_history,                     # <-- PATCH: voll befüllt
        "pheromone_history": [float(x) for x in pher_history_mean],
        "mycel_params": {
            "guidance": float(mycel_guidance_strength),
            "decay": float(mycel_decay),
            "diffusion": float(mycel_diffusion),
            "k_neighbors": int(mycel_k_neighbors),
            "topk_bias": (int(mycel_topk_bias) if mycel_topk_bias else None),
        },
        "vqe_fitness": {
            "enabled": bool(use_vqe_fitness),
            "weight": float(vqe_weight),
            "elite_k": int(vqe_elite_k),
            "num_qubits": int(vqe_num_qubits),
            "layers": int(vqe_layers),
            "cache_size": len(vqe_cache),
        }
    }
    return formulas, table, meta


# ============================================================================
# D) Guidance/Helper
# ============================================================================

def _apply_pheromone_guidance(vec: np.ndarray, pher: np.ndarray, strength: float, topk: int | None = None) -> np.ndarray:
    if strength <= 0.0:
        return vec
    p = np.maximum(pher.astype(np.float32), 0.0)
    if topk and topk < p.size:
        idx = np.argsort(p)[-topk:]
        mask = np.zeros_like(p); mask[idx] = p[idx]
        p = mask
    τ = 0.2
    s = np.exp((p - p.max()) / max(1e-6, τ)).astype(np.float32)
    s = s / (float(s.sum()) + 1e-8)
    guided = (1.0 - strength) * vec + strength * s
    return _renorm_nonneg(guided)
