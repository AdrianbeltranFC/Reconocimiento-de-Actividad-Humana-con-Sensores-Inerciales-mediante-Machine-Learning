"""
interfaz/core/top8.py
Conversión de una ventana a las 8 features (nombres exactos).
"""

import pandas as pd
import numpy as np
from interfaz.core.features import extract_top8_from_signal

ORANGE_TOP8 = [
    "Acceleration X(g)_mean",
    "Acceleration X(g)_std",
    "Acceleration X(g)_var",
    "Acceleration X(g)_median",
    "Acceleration X(g)_iqr",
    "Acceleration X(g)_rms",
    "Acceleration X(g)_ptp",
    "Acceleration X(g)_sma",
]

def window_df_to_top8(window_df: pd.DataFrame, acc_x_col_name: str = 'Acceleration X(g)') -> pd.Series:
    """
    Calcula las 8 características desde una ventana DataFrame y devuelve pandas.Series
    con los nombres exactos ORANGE_TOP8 en ese orden.
    """
    if acc_x_col_name not in window_df.columns:
        raise ValueError(f"La columna {acc_x_col_name} no está presente en la ventana")
    sig = window_df[acc_x_col_name].to_numpy(dtype=float)
    feats = extract_top8_from_signal(sig)
    series = [
        feats['mean'], feats['std'], feats['var'], feats['median'],
        feats['iqr'], feats['rms'], feats['ptp'], feats['sma']
    ]
    return pd.Series(series, index=ORANGE_TOP8)
