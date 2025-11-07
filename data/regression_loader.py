import numpy as np
import pandas as pd
from pathlib import Path

def load_regression_csv(path=None):
    project_root = Path(__file__).resolve().parent.parent
    
    if path is None:
        csv_path = project_root / "data" / "Student_Performance.csv"
    else:
        csv_path = Path(path)
        
    df = pd.read_csv(csv_path, sep=",")
    X = df[["Hours Studied","Previous Scores","Sleep Hours", "Sample Question Papers Practiced"]].values.astype(np.float32)
    X_m = (df["Extracurricular Activities"]=="Yes").values.astype(np.float32)[:,None]
    X_final = np.hstack([X, X_m])
    y = df["Performance Index"].values.astype(np.float32).reshape(-1,1)
    return X_final, y