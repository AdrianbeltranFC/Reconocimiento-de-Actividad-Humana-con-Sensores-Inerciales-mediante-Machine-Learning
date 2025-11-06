# interfaz/core/features.py
"""
Funciones de segmentación y extracción de características.

Contiene:
 - sliding_windows: generador de ventanas por muestras.
 - extract_top8_from_signal: las 8 features originales (mean,std,var,median,iqr,rms,ptp,sma).
 - extract_features_from_signal: versión completa (tiempo + espectrales).
 - detect_channel_columns: heurística para mapear nombres de columnas a canales (acc_x, gyr_x, ang_x...).
"""
from typing import Dict, List, Tuple, Generator
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, iqr, entropy

# ---------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------
def sliding_windows(df: pd.DataFrame, window_samples: int, step_samples: int) -> Generator[Tuple[int,int,pd.DataFrame], None, None]:
    """
    Generador de ventanas: (start_idx, end_idx, window_df)
    - df: DataFrame indexado por entero (0..n-1)
    - window_samples: tamaño de ventana en muestras
    - step_samples: salto en muestras (p.ej. 128 para 50% con 256 ventana)
    """
    n = len(df)
    start = 0
    while start + window_samples <= n:
        end = start + window_samples
        yield start, end, df.iloc[start:end]
        start += step_samples


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def zero_crossing_rate(signal: np.ndarray) -> float:
    s = np.asarray(signal)
    if s.size == 0:
        return 0.0
    return float(np.sum(np.abs(np.diff(np.sign(s)))) / (2 * len(s) + 1e-12))


def spectral_entropy_from_pxx(pxx: np.ndarray) -> float:
    p = pxx.copy()
    denom = (p.sum() + 1e-12)
    p_norm = p / denom
    return float(entropy(p_norm + 1e-12))


# ---------------------------------------------------------------------
# Extracción: top8 (ligera) - compatibilidad con tu modelo KNN_8
# ---------------------------------------------------------------------
def extract_top8_from_signal(sig: np.ndarray) -> Dict[str, float]:
    """
    Extrae las 8 features simples sobre una señal 1D (ej. Acc X).
    Retorna diccionario con llaves: mean,std,var,median,iqr,rms,ptp,sma
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return dict(mean=np.nan, std=np.nan, var=np.nan, median=np.nan, iqr=np.nan, rms=np.nan, ptp=np.nan, sma=np.nan)
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    var = float(np.var(sig))
    median = float(np.median(sig))
    iqr_v = float(_iqr(sig))
    rms = float(np.sqrt(np.mean(sig**2)))
    ptp = float(np.ptp(sig))
    sma = float(np.sum(np.abs(sig)))
    return dict(mean=mean, std=std, var=var, median=median, iqr=iqr_v, rms=rms, ptp=ptp, sma=sma)


# ---------------------------------------------------------------------
# Extracción completa por ventana (tiempo + frecuencia)
# ---------------------------------------------------------------------
def extract_features_from_signal(sig: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extrae features de dominio del tiempo y frecuencia para una señal 1D.
    - Entradas:
        sig: array 1D
        fs: frecuencia (Hz)
    - Salidas (ejemplos, con sufijos):
        mean,std,var,median,iqr,rms,ptp,sma,skew,kurtosis,zcr,
        dom_freq,spec_entropy,total_power
    """
    s = np.asarray(sig, dtype=float)
    feats = {}
    if s.size == 0:
        # rellenar con ceros/NaNs manejables
        zeros = {k: 0.0 for k in ['mean','std','var','median','iqr','rms','ptp','sma','skew','kurtosis','zcr','dom_freq','spec_entropy','total_power']}
        return zeros

    feats['mean'] = float(np.mean(s))
    feats['std'] = float(np.std(s))
    feats['var'] = float(np.var(s))
    feats['median'] = float(np.median(s))
    feats['iqr'] = float(iqr(s))
    feats['rms'] = float(np.sqrt(np.mean(s**2)))
    feats['ptp'] = float(np.ptp(s))
    feats['sma'] = float(np.sum(np.abs(s)) / (len(s) + 1e-12))   # normalizada por número de muestras
    feats['skew'] = float(skew(s))
    feats['kurtosis'] = float(kurtosis(s))
    feats['zcr'] = float(zero_crossing_rate(s))

    # Dominio frecuencia usando Welch
    try:
        f, Pxx = welch(s, fs=fs, nperseg=min(len(s), 256))
        if np.all(Pxx == 0) or np.sum(Pxx) == 0:
            feats['dom_freq'] = 0.0
            feats['spec_entropy'] = 0.0
            feats['total_power'] = 0.0
        else:
            feats['dom_freq'] = float(f[np.argmax(Pxx)])
            feats['spec_entropy'] = float(spectral_entropy_from_pxx(Pxx))
            feats['total_power'] = float(np.trapz(Pxx, f))
    except Exception:
        feats['dom_freq'] = 0.0
        feats['spec_entropy'] = 0.0
        feats['total_power'] = 0.0

    return feats


# ---------------------------------------------------------------------
# Mapeo de columnas detectadas (heurística)
# ---------------------------------------------------------------------
def detect_channel_columns(cols: List[str]) -> Dict[str, str]:
    """
    Dado un listado de columnas, devuelve un mapeo:
      'acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','ang_x','ang_y','ang_z' -> nombre_col
    Usa heurísticas sobre el nombre (case-insensitive).
    """
    colmap = {}
    for c in cols:
        lc = c.lower()
        if "acceleration x" in lc or re_search(lc, r"\baccel.*x\b") or re_search(lc, r"\bax\b"):
            colmap['acc_x'] = c
        if "acceleration y" in lc or re_search(lc, r"\baccel.*y\b") or re_search(lc, r"\bay\b"):
            colmap['acc_y'] = c
        if "acceleration z" in lc or re_search(lc, r"\baccel.*z\b") or re_search(lc, r"\baz\b"):
            colmap['acc_z'] = c
        if "angular velocity x" in lc or re_search(lc, r"\bgyro.*x\b") or re_search(lc, r"\bang.*x\b"):
            colmap['gyr_x'] = c
        if "angular velocity y" in lc or re_search(lc, r"\bgyro.*y\b") or re_search(lc, r"\bang.*y\b"):
            colmap['gyr_y'] = c
        if "angular velocity z" in lc or re_search(lc, r"\bgyro.*z\b") or re_search(lc, r"\bang.*z\b"):
            colmap['gyr_z'] = c
        if "angle x" in lc or re_search(lc, r"\bangle.*x\b"):
            colmap['ang_x'] = c
        if "angle y" in lc or re_search(lc, r"\bangle.*y\b"):
            colmap['ang_y'] = c
        if "angle z" in lc or re_search(lc, r"\bangle.*z\b"):
            colmap['ang_z'] = c

    # heurística: si no detectó por etiquetas largas, buscar patrones 'acc x' o 'ax'
    if not any(k in colmap for k in ('acc_x','acc_y','acc_z')):
        acc_candidates = [c for c in cols if ('acc' in c.lower() and 'x' in c.lower()) or re_search(c.lower(), r'\bax\b')]
        if len(acc_candidates) >= 1:
            colmap['acc_x'] = acc_candidates[0]
    # devolver mapping
    return colmap


# pequeño helper local para evitar importar re repetidamente
import re
def re_search(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None
