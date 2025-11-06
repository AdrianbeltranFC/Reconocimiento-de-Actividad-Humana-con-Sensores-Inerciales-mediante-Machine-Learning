"""
interfaz/core/top8.py
Conversión de una ventana a las 8 features (nombres exactos).
"""

import pandas as pd
import numpy as np
import logging
from interfaz.core.features import extract_top8_from_signal

logger = logging.getLogger(__name__)

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

def validate_features(features_df):
    """Valida rango y distribución de características."""
    for col in features_df.columns:
        values = features_df[col].values
        logger.debug(f"Feature {col}:")
        logger.debug(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
        logger.debug(f"  Mean ± Std: {values.mean():.3f} ± {values.std():.3f}")
        
        # Detectar valores atípicos
        z_scores = np.abs((values - values.mean()) / values.std())
        outliers = z_scores > 3
        if outliers.any():
            logger.warning(f"  Outliers detected in {col}: {sum(outliers)} values")
    
    return features_df

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
