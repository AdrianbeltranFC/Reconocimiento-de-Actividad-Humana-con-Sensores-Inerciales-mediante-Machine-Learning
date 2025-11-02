#!/usr/bin/env python3
"""
03_build_dataset.py
Concatena todos los CSV de features (data/features/.../*_features.csv) en un CSV único:
- Asegura que las columnas de feature sean coherentes
- Añade Sujeto y Clase (si faltan)
- Guarda en data/final/All_features.csv

Uso:
python src/03_build_dataset.py --input_dir data/features --output_file data/final/All_features.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def gather_feature_files(input_dir: str):
    p = Path(input_dir)
    files = list(p.rglob("*_features.csv"))
    return files

def load_and_standardize(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    # Asegurar columnas Sujeto y Clase
    if 'Sujeto' not in df.columns:
        # tratar de inferir desde ruta
        parts = Path(file_path).parts
        # intentar última carpeta antes del archivo para clase y anterior para sujeto
        sujeto = None; clase = None
        if len(parts) >= 3:
            sujeto = parts[-3] if parts[-3].upper().startswith("S") else None
            clase = parts[-2]
        if sujeto is None:
            sujeto = "Unknown"
        if 'Sujeto' not in df.columns:
            df['Sujeto'] = sujeto
        if 'Clase' not in df.columns:
            df['Clase'] = clase if clase is not None else "Desconocida"
    return df

def main():
    parser = argparse.ArgumentParser(description="03_build_dataset - concatena features")
    parser.add_argument("--input_dir", type=str, default="data/features", help="Carpeta con *_features.csv")
    parser.add_argument("--output_file", type=str, default="data/final/All_features.csv", help="CSV final")
    args = parser.parse_args()

    files = gather_feature_files(args.input_dir)
    if len(files) == 0:
        print("No se encontraron archivos de features en", args.input_dir)
        return

    dfs = []
    for f in tqdm(files, desc="Cargando features"):
        try:
            df = load_and_standardize(f)
            dfs.append(df)
        except Exception as e:
            print("Error cargando", f, e)

    df_all = pd.concat(dfs, ignore_index=True)
    # Ordenar columnas: Sujeto, Clase, archivo, start_idx, end_idx, luego features
    cols = list(df_all.columns)
    front = [c for c in ['Sujeto','Clase','archivo','start_idx','end_idx'] if c in cols]
    others = [c for c in cols if c not in front]
    cols_ordered = front + others
    df_all = df_all[cols_ordered]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print("Dataset final guardado en", out_path, " Filas:", len(df_all))

if __name__ == "__main__":
    main()
