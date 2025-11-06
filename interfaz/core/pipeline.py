# interfaz/core/pipeline.py
"""
Pipeline: ventana y extracción top8 usando la resolución del modelo (100 Hz).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from interfaz.core.features import sliding_windows
from interfaz.core.top8 import window_df_to_top8, ORANGE_TOP8

# modelo por defecto
DEFAULT_MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")
MODEL_FS = 100.0  # frecuencia a la que el modelo fue entrenado (según tu comentario)
MODEL_WINDOW_SEC = 2.56  # ventana de entrenamiento
MODEL_OVERLAP = 0.5      # overlap por defecto (si quieres 0.5)

def choose_window_params(fs: float):
    """
    Si fs >= 50 -> ventana 2.56 s, overlap 0.5
    Si 25 <= fs < 50 -> ventana 2.56 s, overlap 0.5 (mantener entrenamiento)
    Si fs < 25 -> ventana 2.56 s, overlap 0.5 (resamplearemos a 100Hz)
    En resumen, para compatibilidad con tu entrenamiento usamos siempre 2.56 s.
    """
    return MODEL_WINDOW_SEC, MODEL_OVERLAP

def run_pipeline_on_processed_df(df_proc: pd.DataFrame,
                                 acc_x_col_name: str = 'Acceleration X(g)',
                                 window_sec: float = None,
                                 overlap: float = None,
                                 model_path: str = None):
    """
    df_proc: DataFrame que idealmente ya está resampleado a MODEL_FS y filtrado;
             si no lo está, asumimos que 'time' existe y procederemos.
    Retorna: df_features (ORANGE_TOP8 + window_center_time), indices, y_pred, y_proba
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    if 'time' not in df_proc.columns:
        raise ValueError("df_proc debe contener columna 'time' en segundos")

    time = df_proc['time'].to_numpy(dtype=float)
    dt = np.diff(time)
    dt = dt[~np.isnan(dt) & (dt>0)]
    if len(dt) > 0:
        fs = float(np.round(1.0/np.median(dt), 6))
    else:
        fs = MODEL_FS  # fuerza fs del modelo si no hay timestamps

    # elegir siempre ventana de entrenamiento salvo que se especifique lo contrario
    if window_sec is None or overlap is None:
        wsec, ov = choose_window_params(fs)
    else:
        wsec, ov = window_sec, overlap

    window_samples = int(round(wsec * fs))
    step = int(round(window_samples * (1.0 - ov)))
    if window_samples <= 0 or step <= 0:
        raise ValueError("Parámetros de ventana inválidos")

    feats_rows = []
    indices = []
    center_times = []

    for start, end, wdf in sliding_windows(df_proc, window_samples, step):
        try:
            s = window_df_to_top8(wdf, acc_x_col_name=acc_x_col_name)
        except Exception:
            s = pd.Series([np.nan]*len(ORANGE_TOP8), index=ORANGE_TOP8)
        feats_rows.append(s)
        indices.append((start, end))
        tcenter = float(np.nanmean(df_proc['time'].iloc[start:end].to_numpy(dtype=float)))
        center_times.append(tcenter)

    if len(feats_rows) == 0:
        raise RuntimeError("No se generaron ventanas.")

    df_feat = pd.DataFrame(feats_rows, columns=ORANGE_TOP8)
    df_feat['window_center_time'] = center_times

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
