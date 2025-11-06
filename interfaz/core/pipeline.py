"""
interfaz/core/pipeline.py 
Pipeline que: toma df procesado (con 'time'), segmenta en ventanas, extrae top8,
 carga el modelo y devuelve features + 
predicciones.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from interfaz.core.features import sliding_windows
from interfaz.core.top8 import window_df_to_top8, ORANGE_TOP8

# Ruta por defecto del modelo (la que pediste)
DEFAULT_MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")

def choose_window_params(fs: float):
    """
    Recomienda (window_sec, overlap) según fs:
      - fs >= 50: 2.56 s, 0.5
      - 25 <= fs < 50: 1.92 s, 0.66
      - fs < 25 (p.ej. 15 Hz): 1.28 s, 0.75
    """
    if fs is None:
        return 2.56, 0.5
    if fs >= 50.0:
        return 2.56, 0.5
    if fs >= 25.0:
        return 1.92, 0.66
    return 1.28, 0.75

def run_pipeline_on_processed_df(df_proc: pd.DataFrame,
                                 acc_x_col_name: str = 'Acceleration X(g)',
                                 window_sec: float = None,
                                 overlap: float = None,
                                 model_path: str = None):
    """
    Ejecuta pipeline:
    - df_proc: DataFrame procesado (de preprocess_csv) con columna 'time' en segundos
    - retorna: (df_features, indices, y_pred, y_proba)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    if 'time' not in df_proc.columns:
        raise ValueError("df_proc debe contener la columna 'time' en segundos")

    time = df_proc['time'].to_numpy(dtype=float)
    dt = np.diff(time)
    dt = dt[~np.isnan(dt) & (dt > 0)]
    fs = 50.0
    if len(dt) > 0:
        fs = float(np.round(1.0 / np.median(dt), 6))

    if window_sec is None or overlap is None:
        wsec, ov = choose_window_params(fs)
    else:
        wsec, ov = window_sec, overlap

    window_samples = int(round(wsec * fs))
    step = int(round(window_samples * (1.0 - ov)))
    if window_samples <= 0 or step <= 0:
        raise ValueError("Parámetros de ventana inválidos (window_samples o step <= 0)")

    feats_rows = []
    indices = []
    for start, end, wdf in sliding_windows(df_proc, window_samples, step):
        try:
            s = window_df_to_top8(wdf, acc_x_col_name=acc_x_col_name)
        except Exception:
            s = pd.Series([np.nan] * len(ORANGE_TOP8), index=ORANGE_TOP8)
        feats_rows.append(s)
        indices.append((start, end))

    if len(feats_rows) == 0:
        raise RuntimeError("No se generaron ventanas. Archivo demasiado corto para la ventana escogida.")

    df_feat = pd.DataFrame(feats_rows)

    # cargar modelo
    model = joblib.load(model_path)
    X = df_feat[ORANGE_TOP8].values
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            try:
                final = list(model.named_steps.values())[-1]
                y_proba = final.predict_proba(X)
            except Exception:
                y_proba = None

    return df_feat, indices, y_pred, y_proba
