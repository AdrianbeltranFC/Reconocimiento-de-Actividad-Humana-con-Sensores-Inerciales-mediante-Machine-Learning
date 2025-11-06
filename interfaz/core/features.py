# -------------------------------
# ARCHIVO: interfaz/core/features.py
# -------------------------------
"""
Utilidades para segmentación en ventanas y extracción de las 8 características
"""
import numpy as np
import pandas as pd


def sliding_windows(df: pd.DataFrame, window_samples: int, step_samples: int):
    """Generador de ventanas: retorna (start_idx, end_idx, window_df)"""
    n = len(df)
    start = 0
    while start + window_samples <= n:
        end = start + window_samples
        yield start, end, df.iloc[start:end]
        start += step_samples


def _iqr(x: np.ndarray):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def extract_top8_from_signal(sig: np.ndarray):
    """Calcula las 8 características principales sobre una señal 1D (Acc X en g).

    Características:
    - mean, std, var, median, iqr, rms, ptp (peak-to-peak), sma (sum of abs)
    """
    if len(sig) == 0:
        return {k: np.nan for k in ['mean','std','var','median','iqr','rms','ptp','sma']}
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    var = float(np.var(sig))
    median = float(np.median(sig))
    iqr = float(_iqr(sig))
    rms = float(np.sqrt(np.mean(sig**2)))
    ptp = float(np.ptp(sig))
    sma = float(np.sum(np.abs(sig)))
    return dict(mean=mean, std=std, var=var, median=median, iqr=iqr, rms=rms, ptp=ptp, sma=sma)