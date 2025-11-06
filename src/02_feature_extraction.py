#!/usr/bin/env python3
"""
02_feature_extraction.py
Lee los CSV procesados en data/processed/.../*_pipeline.csv
- Crea ventanas: 2.56 s con un solapamiento del 50%
- fs por defecto 100 Hz (dado como se resampleó en preprocessing).
- Extrae features por ventana para cada canal detectado (acel, gyro, ang)
- Guarda archivos por sujeto/clase en data/features/<Sujeto>/<Clase>/<archivo>_features.csv

Uso:
python src/02_feature_extraction.py --input_dir data/processed --output_dir data/features
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, iqr, entropy
from tqdm import tqdm

# ------------------------------
# Funciones de extracción
# ------------------------------
def detect_channel_columns(cols: List[str]) -> Dict[str, str]:
    """Devuelve mapping de canales detectados a nombres de columna."""
    colmap = {}
    for c in cols:
        lc = c.lower()
        if "acceleration x" in lc or re.search(r"\baccel.*x\b", lc) or re.search(r"\bacc.*x\b", lc):
            colmap['acc_x'] = c
        if "acceleration y" in lc or re.search(r"\baccel.*y\b", lc):
            colmap['acc_y'] = c
        if "acceleration z" in lc or re.search(r"\baccel.*z\b", lc):
            colmap['acc_z'] = c

        if "angular velocity x" in lc or re.search(r"\bangular.*x\b", lc) or re.search(r"\bgyro.*x\b", lc):
            colmap['gyr_x'] = c
        if "angular velocity y" in lc or re.search(r"\bangular.*y\b", lc) or re.search(r"\bgyro.*y\b", lc):
            colmap['gyr_y'] = c
        if "angular velocity z" in lc or re.search(r"\bangular.*z\b", lc) or re.search(r"\bgyro.*z\b", lc):
            colmap['gyr_z'] = c

        if "angle x" in lc:
            colmap['ang_x'] = c
        if "angle y" in lc:
            colmap['ang_y'] = c
        if "angle z" in lc:
            colmap['ang_z'] = c

  
    accel_candidates = [c for c in cols if 'accel' in c.lower() or 'acceleration' in c.lower()]
    if not any(k in colmap for k in ('acc_x','acc_y','acc_z')) and len(accel_candidates) >= 3:
        colmap['acc_x'], colmap['acc_y'], colmap['acc_z'] = accel_candidates[:3]
    return colmap

def spectral_entropy(pxx):
    """Entropía espectral de densidad de potencia normalizada."""
    p = pxx.copy()
    p_norm = p / (p.sum() + 1e-12)
    return entropy(p_norm + 1e-12)

def zero_crossing_rate(signal):
    s = np.asarray(signal)
    # crossings count / length
    return np.sum(np.abs(np.diff(np.sign(s)))) / (2 * len(s) + 1e-12)

def extract_features_from_signal(sig: np.ndarray, fs: float):
    """
    Devuelve dict de features para una señal 1D:
    - dominio del tiempo: mean, std, var, median, iqr, rms, ptp, sma, skew, kurtosis, zcr
    - dominio frecuencia: dom_freq, spec_entropy, total_power
    """
    s = np.asarray(sig, dtype=float)
    feats = {}
    # tiempo
    feats['mean'] = float(np.mean(s))
    feats['std'] = float(np.std(s))
    feats['var'] = float(np.var(s))
    feats['median'] = float(np.median(s))
    feats['iqr'] = float(iqr(s))
    feats['rms'] = float(np.sqrt(np.mean(s**2)))
    feats['ptp'] = float(np.ptp(s))
    feats['sma'] = float(np.sum(np.abs(s)) / (len(s) + 1e-12))
    feats['skew'] = float(skew(s))
    feats['kurtosis'] = float(kurtosis(s))
    feats['zcr'] = float(zero_crossing_rate(s))

    # frecuencia (Welch)
    try:
        f, Pxx = welch(s, fs=fs, nperseg=min(len(s), 256))
        if np.all(Pxx == 0):
            feats['dom_freq'] = 0.0
            feats['spec_entropy'] = 0.0
            feats['total_power'] = 0.0
        else:
            feats['dom_freq'] = float(f[np.argmax(Pxx)])
            feats['spec_entropy'] = float(spectral_entropy(Pxx))
            feats['total_power'] = float(np.trapz(Pxx, f))
    except Exception:
        feats['dom_freq'] = 0.0
        feats['spec_entropy'] = 0.0
        feats['total_power'] = 0.0

    return feats

# ------------------------------
# Pipeline: procesar un archivo _pipeline.csv
# ------------------------------
def process_processed_file(filepath: str, output_dir: str, fs: float = 100.0,
                           window_sec: float = 2.56, overlap: float = 0.5, verbose: bool=False):
    df = pd.read_csv(filepath, low_memory=False)
    cols = list(df.columns)
    colmap = detect_channel_columns(cols)

    # Requerimos al menos una señal para extraer
    if len(colmap) == 0:
        if verbose:
            print("No se detectaron canales en", filepath)
        return None

    # Ventana y paso
    n_win = int(round(window_sec * fs))           # 2.56 * 100 = 256
    step = int(round(n_win * (1 - overlap)))      # 50% -> 128

    if n_win <= 1 or step < 1:
        raise ValueError("Parámetros de ventana inválidos: n_win=%s step=%s" % (n_win, step))

    features_rows = []

    # construir etiqueta (Sujeto, Clase) desde ruta/filename
    fname = os.path.basename(filepath)
    sujeto = "Unknown"
    clase = "Desconocida"
    m_s = re.search(r"(S\d{1,2})", fname, re.IGNORECASE)
    if m_s:
        sujeto = m_s.group(1)
    if re.search(r'quieto', fname, re.IGNORECASE):
        clase = "Quieto"
    elif re.search(r'caminar', fname, re.IGNORECASE):
        clase = "Caminar"
    elif re.search(r'correr', fname, re.IGNORECASE):
        clase = "Correr"

    # iterar ventanas
    n_rows = len(df)
    for start in range(0, n_rows - n_win + 1, step):
        end = start + n_win
        row = {'Sujeto': sujeto, 'Clase': clase, 'archivo': fname, 'start_idx': start, 'end_idx': end}
        # para cada canal, extraer features y añadir prefix
        for key, colname in colmap.items():
            sig = pd.to_numeric(df[colname], errors='coerce').fillna(method='ffill').fillna(method='bfill').to_numpy()
            window_sig = sig[start:end]
            feats = extract_features_from_signal(window_sig, fs=fs)
            for k, v in feats.items():
                row[f"{colname}_{k}"] = v
        features_rows.append(row)

    if len(features_rows) == 0:
        return None

    df_features = pd.DataFrame(features_rows)

    # guardar en estructura salida /Sujeto/Clase/
    out_folder = Path(output_dir) / sujeto / clase
    out_folder.mkdir(parents=True, exist_ok=True)
    out_name = fname.replace("_pipeline.csv", "_features.csv")
    out_path = out_folder / out_name
    df_features.to_csv(out_path, index=False)
    return str(out_path)

# ------------------------------
# CLI para procesar carpeta completa
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="02_feature_extraction - extrae features por ventana")
    parser.add_argument("--input_dir", type=str, default="data/processed", help="Carpeta con *_pipeline.csv")
    parser.add_argument("--output_dir", type=str, default="data/features", help="Carpeta de salida para features")
    parser.add_argument("--fs", type=float, default=100.0, help="Frecuencia (Hz)")
    parser.add_argument("--window_sec", type=float, default=2.56, help="Duración de ventana (s)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Solapamiento (0-1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    csv_files = list(in_dir.rglob("*_pipeline.csv"))
    if len(csv_files) == 0:
        print("No se encontraron archivos *_pipeline.csv en", in_dir)
        return

    resultados = []
    for f in tqdm(csv_files, desc="Extrayendo features"):
        try:
            out_path = process_processed_file(str(f), str(out_dir), fs=args.fs, window_sec=args.window_sec, overlap=args.overlap, verbose=args.verbose)
            resultados.append((str(f), out_path))
        except Exception as e:
            resultados.append((str(f), None))
            if args.verbose:
                print("Error en", f, e)

    print("Extracción finalizada. Archivos procesados:", sum(1 for _,o in resultados if o is not None))

if __name__ == "__main__":
    main()
