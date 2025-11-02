#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_finalize_models_and_reports.py

- Re-evalúa baseline vs Orange-top8 (StratifiedKFold por defecto)
- Guarda matrices de confusión y classification reports (CSV + PNG) en español
- Entrena modelos finales (SVM_RBF y kNN) sobre la matriz reducida y los guarda en models/
- Guarda tabla comparativa con métricas en reports/final_models/final_comparison_detailed.csv

Uso:
python src/07_finalize_models_and_reports.py --input_csv data/final/All_features.csv --reduced_csv data/final/All_features_orange_top8.csv --cv stratified --n_splits 5 --save_models --verbose
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# ---------------------------
# Utilidades
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def printv(msg, verbose):
    if verbose:
        print(msg)

# ---------------------------
# Funciones de plotting (español)
# ---------------------------
def plot_confusion_matrix(cm, labels, out_png: Path, title="Matriz de confusión"):
    """Guarda matriz de confusión como imagen (con anotaciones)."""
    ensure_dir(out_png.parent)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_classification_report(y_true, y_pred, labels, out_csv: Path):
    """Genera classification report (precision/recall/f1) y lo guarda como CSV."""
    cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_cr = pd.DataFrame(cr).T
    df_cr.to_csv(out_csv)
    return df_cr

# ---------------------------
# Evaluación por CV (predicciones agregadas)
# ---------------------------
def cv_predict_and_report(X, y, pipeline, cv, labels, out_prefix: Path, verbose=False):
    """
    Ejecuta cross_val_predict para obtener predicciones agregadas (por CV),
    calcula matriz de confusión, classification report y guarda todo.
    """
    # obtén predicciones por CV (respetando imputación dentro de pipeline)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method='predict', n_jobs=-1)
    # métricas generales
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
    # confusion matrix
    cm = confusion_matrix(y, y_pred, labels=labels)
    # guardar cm imagen
    plot_confusion_matrix(cm, labels, out_prefix.with_suffix('.png'), title=out_prefix.name)
    # classification report CSV
    cr_df = save_classification_report(y, y_pred, labels, out_prefix.with_suffix('.csv'))
    if verbose:
        print(f"[CV] {out_prefix.name} — accuracy: {acc:.6f}, f1_macro: {f1:.6f}")
    return {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'confusion_matrix': cm,
        'classification_report_df': cr_df
    }

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrena y guarda modelos finales + reportes (matrices de confusión y classification reports) en español.")
    parser.add_argument("--input_csv", type=str, default="data/final/All_features.csv", help="CSV completo (baseline).")
    parser.add_argument("--reduced_csv", type=str, default="data/final/All_features_orange_top8.csv", help="CSV reducido (Orange top8).")
    parser.add_argument("--target_col", type=str, default="Clase")
    parser.add_argument("--cv", choices=['stratified'], default='stratified', help="Tipo de CV (solo stratified implementado).")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_models", action="store_true", help="Si se pasa, guarda modelos finales entrenados sobre la matriz reducida.")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports/final_models")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    inp = Path(args.input_csv)
    red = Path(args.reduced_csv)
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    ensure_dir(reports_dir); ensure_dir(models_dir)

    if not inp.exists():
        raise FileNotFoundError(f"No existe {inp}")
    if not red.exists():
        raise FileNotFoundError(f"No existe {red}")

    # Cargar CSVs
    df_full = pd.read_csv(inp, low_memory=False)
    df_red = pd.read_csv(red, low_memory=False)

    if args.target_col not in df_full.columns:
        raise ValueError(f"No existe columna target '{args.target_col}' en {inp}")

    # etiquetas/labels ordenadas
    y_full = df_full[args.target_col].astype(str)
    labels = sorted(y_full.unique())

    # Baseline: usar todas las columnas numéricas (excluir metadata si existen)
    metadata = [c for c in ['Sujeto','Clase','archivo','start_idx','end_idx'] if c in df_full.columns]
    X_full = df_full.drop(columns=[c for c in metadata if c in df_full.columns], errors='ignore')
    X_full = X_full.select_dtypes(include=[np.number])
    printv(f"Baseline: {X_full.shape[1]} features numéricas", args.verbose)
    # Reduced: cargar X_red (asegurar que están las columnas numéricas)
    meta_red = [c for c in ['Sujeto','Clase','archivo','start_idx','end_idx'] if c in df_red.columns]
    X_red = df_red.drop(columns=[c for c in meta_red if c in df_red.columns], errors='ignore')
    X_red = X_red.select_dtypes(include=[np.number])
    printv(f"Reducida: {X_red.shape[1]} features (Orange top8)", args.verbose)

    # Crear CV
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # Pipelines con imputación por mediana + escalado
    svm_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    knn_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    # --- 1) Evaluación baseline (todas las features) ---
    print("Evaluando BASELINE (todas las features numéricas) con CV...")
    res_baseline_svm = cv_predict_and_report(X_full, y_full, svm_pipe, cv, labels, reports_dir / "baseline_SVM_confmat", verbose=args.verbose)
    res_baseline_knn = cv_predict_and_report(X_full, y_full, knn_pipe, cv, labels, reports_dir / "baseline_kNN_confmat", verbose=args.verbose)

    # --- 2) Evaluación reducida (Orange top8) ---
    print("Evaluando REDUCIDA (Orange top8) con CV...")
    y_red = df_red[args.target_col].astype(str)
    res_reduced_svm = cv_predict_and_report(X_red, y_red, svm_pipe, cv, labels, reports_dir / "reduced_SVM_confmat", verbose=args.verbose)
    res_reduced_knn = cv_predict_and_report(X_red, y_red, knn_pipe, cv, labels, reports_dir / "reduced_kNN_confmat", verbose=args.verbose)

    # --- 3) Guardar resumen comparativo (CSV con métricas agregadas) ---
    rows = []
    rows.append({
        'model': 'SVM_RBF',
        'baseline_acc_mean': res_baseline_svm['accuracy'],
        'baseline_f1_macro': res_baseline_svm['f1_macro'],
        'reduced_acc_mean': res_reduced_svm['accuracy'],
        'reduced_f1_macro': res_reduced_svm['f1_macro']
    })
    rows.append({
        'model': 'kNN_k5',
        'baseline_acc_mean': res_baseline_knn['accuracy'],
        'baseline_f1_macro': res_baseline_knn['f1_macro'],
        'reduced_acc_mean': res_reduced_knn['accuracy'],
        'reduced_f1_macro': res_reduced_knn['f1_macro']
    })
    df_comp = pd.DataFrame(rows)
    comp_csv = reports_dir / "final_models_comparison_summary.csv"
    df_comp.to_csv(comp_csv, index=False)
    print("Resumen comparativo guardado en:", comp_csv)

    # --- 4) Entrenar modelos finales sobre toda la matriz reducida y guardarlos (opcional si --save_models) ---
    if args.save_models:
        print("Entrenando modelos finales sobre la matriz REDUCIDA y guardando en 'models/' ...")
        # entrenar con todo X_red, y_red
        svm_pipe.fit(X_red, y_red)
        knn_pipe.fit(X_red, y_red)
        joblib.dump(svm_pipe, models_dir / "SVM_orange_top8_final.joblib")
        joblib.dump(knn_pipe, models_dir / "kNN_orange_top8_final.joblib")
        print("Modelos guardados en:", models_dir)

    # --- 5) Guardar classification reports por modelo (CSV ya generado en cv_predict_and_report) ---
    # (ya los guardamos como CSV al crear las figuras)

    # --- 6) Guardar también las matrices de confusión numéricas para el reporte (CSV) ---
    # Baseline SVM
    pd.DataFrame(res_baseline_svm['confusion_matrix'], index=labels, columns=labels).to_csv(reports_dir / "baseline_SVM_confusion_matrix.csv")
    pd.DataFrame(res_baseline_knn['confusion_matrix'], index=labels, columns=labels).to_csv(reports_dir / "baseline_kNN_confusion_matrix.csv")
    pd.DataFrame(res_reduced_svm['confusion_matrix'], index=labels, columns=labels).to_csv(reports_dir / "reduced_SVM_confusion_matrix.csv")
    pd.DataFrame(res_reduced_knn['confusion_matrix'], index=labels, columns=labels).to_csv(reports_dir / "reduced_kNN_confusion_matrix.csv")

    t1 = time.time()
    print("\nProceso completado en {:.1f} s.".format(t1 - t0))
    print("Revisa la carpeta:", reports_dir, "y models (si pediste guardar).")

if __name__ == "__main__":
    main()
