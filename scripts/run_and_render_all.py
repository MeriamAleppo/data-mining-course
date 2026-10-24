"""
Script to build and execute all three notebooks in order, pre-rendering all outputs.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import nbformat as nbf
from nbclient import NotebookClient

from scripts.build_nb1 import create_notebook_1
from scripts.build_nb2 import create_notebook_2
from scripts.build_nb3 import create_notebook_3

notebooks_config = [
    ("01_data_ingestion_and_cleaning.ipynb", create_notebook_1),
    ("02_eda_and_feature_engineering.ipynb", create_notebook_2),
    ("03_hypothesis_testing_and_modeling.ipynb", create_notebook_3),
]

def run_notebook(nb_path: Path, nb_obj):
    print(f"\n==========================================")
    print(f"Generating & Executing: {nb_path.name}")
    print(f"==========================================")
    start_time = time.time()

    client = NotebookClient(
        nb_obj,
        timeout=600,
        kernel_name="python3",
        allow_errors=False,
    )

    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb_obj, f)

    elapsed = time.time() - start_time
    print(f"Successfully executed & rendered {nb_path.name} in {elapsed:.2f} seconds.")

def main():
    for filename, builder_func in notebooks_config:
        nb_path = Path(filename)
        nb_obj = builder_func()
        run_notebook(nb_path, nb_obj)
    print("\nALL 3 NOTEBOOKS GENERATED AND PRE-RENDERED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
