# check_units.py
import numpy as np
import pandas as pd
from pathlib import Path
from interfaz.core.preprocessing import preprocess_csv_for_model

examples = [
    "data/raw/S01_Caminar_1.csv",
    "data/raw/S01_Quieto_1.csv",
    "data/raw/S06_Correr_1.csv"
]
for f in examples:
    df_proc, _ = preprocess_csv_for_model(f, output_dir="interfaz_tmp_processed", target_fs=100.0)
    if 'Acceleration X(g)' in df_proc.columns:
        arr = df_proc['Acceleration X(g)'].to_numpy()
    else:
        # buscar una columna que parezca accel
        candidates = [c for c in df_proc.columns if 'acc' in c.lower()]
        arr = df_proc[candidates[0]].to_numpy()
    print(f"Archivo {Path(f).name}: mean={np.nanmean(arr):.3g}, std={np.nanstd(arr):.3g}, min={np.nanmin(arr):.3g}, max={np.nanmax(arr):.3g}")
    # RMS
    rms = np.sqrt(np.nanmean(arr**2))
    print("  RMS:", rms)
    # si RMS ~ 9-10 -> m/s^2. Si ~1 -> g. Si ~0.01 -> probablemente unidades muy pequeñas.
