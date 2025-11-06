# test_predict_features.py
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from interfaz.core.preprocessing import preprocess_csv_for_model
from interfaz.core.pipeline import run_pipeline_on_processed_df
MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")

m = joblib.load(MODEL_PATH)
print("Cargado modelo:", type(m))

csv_ejemplo = "data/raw/S01_Caminar_1.csv"  # pon aquí un archivo de ejemplo
df_proc, meta = preprocess_csv_for_model(csv_ejemplo, output_dir="interfaz_tmp_processed", target_fs=100.0)
print("df_proc columnas:", df_proc.columns.tolist())
df_feat, indices, y_pred_vals, y_proba_vals = run_pipeline_on_processed_df(df_proc)
print("df_feat columnas:", df_feat.columns.tolist())
print("Primeras filas de df_feat:")
print(df_feat.head())

# PRUEBA A: pasar numpy array (lo que hace la GUI a veces)
X_array = df_feat[[c for c in df_feat.columns if c != 'window_center_time']].values
try:
    preds_a = m.predict(X_array)
    print("Preds con numpy array OK (len):", len(preds_a))
except Exception as e:
    print("Error predict con numpy array:", e)

# PRUEBA B: pasar DataFrame (con nombres). Si modelo define feature_names_in_, reordena:
if hasattr(m, 'feature_names_in_'):
    expected = list(m.feature_names_in_)
    print("Modelo espera columnas:", expected)
    # generar df con exactamente esas columnas (si están en df_feat)
    if all([c in df_feat.columns for c in expected]):
        X_df = df_feat[expected]
        try:
            preds_b = m.predict(X_df)
            print("Preds con DataFrame reordenado OK (len):", len(preds_b))
        except Exception as e:
            print("Error predict con DataFrame reordenado:", e)
    else:
        missing = [c for c in expected if c not in df_feat.columns]
        print("Faltan columnas esperadas por el modelo en df_feat:", missing)
else:
    print("Modelo no define feature_names_in_.")
