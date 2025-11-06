"""
05_ML_first_model.py
Entrena y evalúa modelos (SVM-RBF y k-NN) sobre la matriz de features.

Salida:
- reportes en reports/train/: métricas CSV y figuras de matriz de confusión
- modelos guardados en models/

Uso:
python src/05_ML_first_model.py --input_csv "data/final/All_features.csv" --target_col Clase --group_col Sujeto --cv stratified --n_splits 5"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_data(path, target_col="Clase", drop_cols=None):
    df = pd.read_csv(path, low_memory=False)
    if drop_cols is None:
        drop_cols = ['Sujeto', 'Clase', 'archivo', 'start_idx', 'end_idx']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(str)
    groups = df['Sujeto'] if 'Sujeto' in df.columns else None
    return X, y, groups, df

def run_stratified_cv(X, y, pipeline, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ['accuracy']
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_estimator=True, n_jobs=1)
    return results

def run_loso(X, y, groups, pipeline):
    logo = LeaveOneGroupOut()
    accs = []
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in tqdm(logo.split(X, y, groups), desc="LOSO folds"):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(Xtr, ytr)
        yp = pipeline.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        y_true_all.extend(list(yte))
        y_pred_all.extend(list(yp))
    avg_acc = np.mean(accs)
    return avg_acc, y_true_all, y_pred_all

def save_confusion(y_true, y_pred, out_path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def summarize(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    macro_f1 = np.mean(f1)
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": dict(zip(labels, zip(prec, rec, f1)))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/final/All_features.csv")
    parser.add_argument("--target_col", type=str, default="Clase")
    parser.add_argument("--group_col", type=str, default="Sujeto")
    parser.add_argument("--cv", type=str, choices=["stratified","loso"], default="stratified")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="reports/train")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    moddir = Path(args.models_dir); moddir.mkdir(parents=True, exist_ok=True)

    X, y, groups, raw_df = load_data(args.input_csv, target_col=args.target_col)
    labels = sorted(y.unique().tolist())

    # Remover columnas con NA o no numéricas en X
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0.0)

    print("Features shape:", X.shape, "Classes:", labels)
    # Modelos a evaluar
    models = {
        "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel='rbf', probability=False, C=1.0, gamma='scale'))]),
        "kNN_k5": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
    }

    summary_rows = []
    for name, pipe in models.items():
        print("\nEvaluando:", name)
        if args.cv == "stratified":
            results = run_stratified_cv(X, y, pipe, n_splits=args.n_splits)
            accs = results['test_accuracy']
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Stratified {args.n_splits}-fold Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            # entrenamos sobre todo y guardamos
            pipe.fit(X, y)
            joblib.dump(pipe, moddir / f"{name}_full.joblib")
            # predicción completa
            y_pred = pipe.predict(X)
            cm_path = out / f"{name}_confmat_whole.png"
            save_confusion(y, y_pred, cm_path, labels=labels)
            metrics = summarize(y, y_pred, labels)
            metrics.update({"cv_mean_acc": mean_acc, "cv_std_acc": std_acc})
            summary_rows.append((name, metrics))
        else:
            # LOSO: groups required
            if groups is None:
                print("No group column (Sujeto) found; cannot do LOSO. Skipping.")
                continue
            avg_acc, y_true_all, y_pred_all = run_loso(X, y, groups, pipe)
            print(f"LOSO average accuracy: {avg_acc:.3f}")
            # save confusion
            cm_path = out / f"{name}_confmat_loso.png"
            save_confusion(y_true_all, y_pred_all, cm_path, labels=labels)
            metrics = summarize(y_true_all, y_pred_all, labels)
            metrics.update({"loso_acc": avg_acc})
            # fit final
            pipe.fit(X, y)
            joblib.dump(pipe, moddir / f"{name}_full.joblib")
            summary_rows.append((name, metrics))

    # Guardar resumen en CSV
    rows = []
    for name, metrics in summary_rows:
        base = {"model": name}
        base.update({k: v for k, v in metrics.items() if k in ['cv_mean_acc','cv_std_acc','loso_acc','accuracy','macro_f1']})
        rows.append(base)
    pd.DataFrame(rows).to_csv(Path(args.output_dir)/"models_summary.csv", index=False)
    print("Resultados guardados en", args.output_dir)

if __name__ == "__main__":
    main()
