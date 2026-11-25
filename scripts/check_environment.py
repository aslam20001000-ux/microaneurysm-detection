"""
Verify that the interpreter is the project venv and that every library imports.
Run:  python scripts/check_environment.py
"""
import importlib
import sys

LIBRARIES = {
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "pandas": "Pandas",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "scipy": "SciPy",
    "PIL": "Pillow",
}

print(f"Python     : {sys.version.split()[0]}")
print(f"Interpreter: {sys.executable}")
in_venv = sys.prefix != sys.base_prefix
print(f"Inside venv: {in_venv}")

failed = []
for module_name, label in LIBRARIES.items():
    try:
        module = importlib.import_module(module_name)
        print(f"  [OK]   {label:<13} {getattr(module, '__version__', '?')}")
    except ImportError as error:
        failed.append(label)
        print(f"  [FAIL] {label:<13} {error}")

print("\nAll libraries OK." if not failed else f"\nMissing: {failed}")
sys.exit(1 if failed or not in_venv else 0)
