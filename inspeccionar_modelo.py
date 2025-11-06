# inspeccionar_modelo.py
import joblib
from pathlib import Path
import pprint

MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")
m = joblib.load(MODEL_PATH)
print("Tipo del objeto cargado:", type(m))
info = {}
info['is_pipeline'] = hasattr(m, 'named_steps')
pprint.pprint(info)

if info['is_pipeline']:
    print("Pipeline steps:", list(m.named_steps.keys()))
# ¿Tiene scaler / imputador?
steps = list(m.named_steps.values()) if info['is_pipeline'] else [m]
for s in steps:
    print("Step type:", type(s))
    if hasattr(s, '__class__'):
        print("  class:", s.__class__.__name__)
# clases y predict_proba
est = list(m.named_steps.values())[-1] if info['is_pipeline'] else m
print("Estimador final:", type(est))
print("Tiene predict_proba:", hasattr(est, 'predict_proba'))
print("Clases (si definidas):", getattr(est, 'classes_', None))
# feature names in (si existe)
if hasattr(m, 'feature_names_in_'):
    print("Modelo guarda feature_names_in_ (qué nombres espera):")
    print(m.feature_names_in_)
else:
    print("El modelo NO define feature_names_in_. Revisa pipeline para ver si usa ColumnTransformer.")
