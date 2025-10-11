import os
import joblib
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Si tu pipeline serializado usa funciones definidas en notebook, crea app/preprocess.py
# con las funciones (mismo nombre) y déjalo importado:
try:
    import app.preprocess  # noqa: F401
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "docs", "modelo.joblib"))

# Nombre oficial de la columna del CSV original:
FEATURE_NAME = os.getenv("FEATURE_NAME", "Textos_espanol")
# Aliases aceptados por robustez:
FEATURE_ALIASES = [FEATURE_NAME, "textos"]

class ModelManager:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def _extract_feature_series(self, df: pd.DataFrame) -> pd.Series:
        # Intenta respetar el nombre del CSV; acepta alias
        for col in FEATURE_ALIASES:
            if col in df.columns and df[col].notna().any():
                return df[col].fillna("").astype(str)
        raise ValueError(f"No se encontró la columna de texto. Se esperaba '{FEATURE_NAME}' (o alias {FEATURE_ALIASES}).")

    def predict(self, instances: List[Dict]) -> List[str]:
        df = pd.DataFrame(instances)
        X = self._extract_feature_series(df)
        preds = self.pipeline.predict(X)
        # Asegura str por si son numéricas (1/3/4)
        return [str(p) for p in preds]

    def retrain(self, instances: List[Dict], validation_ratio: float = 0.2, random_state: int = 42) -> Dict:
        df = pd.DataFrame(instances)
        if "label" not in df.columns:
            raise ValueError("Falta la columna 'label' en las instancias para reentrenamiento.")
        X = self._extract_feature_series(df)
        y = df["label"].astype(str)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=validation_ratio, random_state=random_state, stratify=y
        )

        self.pipeline.fit(X_tr, y_tr)
        joblib.dump(self.pipeline, self.model_path)

        y_pred = self.pipeline.predict(X_va)
        pm, rm, fm, _ = precision_recall_fscore_support(y_va, y_pred, average="macro")
        pr, rr, fr, sup = precision_recall_fscore_support(y_va, y_pred, average=None, labels=sorted(y_va.unique()))
        per_class = {
            str(lbl): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lbl, p, r, f, s in zip(sorted(y_va.unique()), pr, rr, fr, sup)
        }

        return {
            "precision_macro": float(pm),
            "recall_macro": float(rm),
            "f1_macro": float(fm),
            "per_class": per_class
        }
