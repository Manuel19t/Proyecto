import numpy as np
import pandas as pd
from pathlib import Path

SPECIES_TO_ID = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def load_iris_csv(path=None):
    # base del proyecto = carpeta padre de data/
    project_root = Path(__file__).resolve().parent.parent

    if path is None:
        csv_path = project_root / "data" / "IRIS.csv"
    else:
        csv_path = Path(path)

    df = pd.read_csv(csv_path)
    X = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy(np.float32)
    y = df["species"].map(SPECIES_TO_ID).to_numpy(dtype=int)
    return X, y, CLASS_NAMES
