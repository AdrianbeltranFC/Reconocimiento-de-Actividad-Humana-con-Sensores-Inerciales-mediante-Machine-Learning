# clean_pycaches.py
import os
import shutil
from pathlib import Path

root = Path(".").resolve()
removed_dirs = 0
removed_files = 0

for p in root.rglob("__pycache__"):
    try:
        shutil.rmtree(p)
        removed_dirs += 1
    except Exception as e:
        print("No pude borrar", p, e)

for p in root.rglob("*.pyc"):
    try:
        p.unlink()
        removed_files += 1
    except Exception as e:
        print("No pude borrar", p, e)

print(f"Directorios __pycache__ eliminados: {removed_dirs}")
print(f"Archivos .pyc eliminados: {removed_files}")
