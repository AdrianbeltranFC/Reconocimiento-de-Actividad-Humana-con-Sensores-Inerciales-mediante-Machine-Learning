#!/usr/bin/env python3
"""
main.py
Orquesta:
1) (opcional) src/01_preprocessing.py
2) src/02_feature_extraction.py
3) src/03_build_dataset.py

Uso:
python src/main.py --run_preprocessing True
"""
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable

def run_cmd(cmd_list):
    print(">", " ".join(cmd_list))
    res = subprocess.run(cmd_list, check=False)
    if res.returncode != 0:
        print("Comando finalizó con código", res.returncode)
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="Orquestador pipeline")
    parser.add_argument("--run_preprocessing", action="store_true", help="Ejecutar preprocessing (src/01_preprocessing.py)")
    parser.add_argument("--input_raw", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--final_csv", type=str, default="data/final/All_features.csv")
    args = parser.parse_args()

    if args.run_preprocessing:
        cmd = [PY, "src/01_preprocessing.py", "--input_dir", args.input_raw, "--output_dir", args.processed_dir, "--verbose"]
        rc = run_cmd(cmd)
        if rc != 0:
            print("Preprocessing falló o finalizó con errores, abortando.")
            return

    # Step 2: feature extraction
    cmd2 = [PY, "src/02_feature_extraction.py", "--input_dir", args.processed_dir, "--output_dir", args.features_dir, "--verbose"]
    rc2 = run_cmd(cmd2)
    if rc2 != 0:
        print("Feature extraction falló o finalizó con errores, revisa logs.")
        return

    # Step 3: build dataset
    cmd3 = [PY, "src/03_build_dataset.py", "--input_dir", args.features_dir, "--output_file", args.final_csv]
    rc3 = run_cmd(cmd3)
    if rc3 != 0:
        print("Build dataset falló o finalizó con errores.")
        return

    print("Pipeline finalizado correctamente. CSV final:", args.final_csv)

if __name__ == "__main__":
    main()
