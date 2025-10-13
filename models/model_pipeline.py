# models/model_pipeline.py
import re, os, joblib, numpy as np, pandas as pd
from typing import Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, classification_report

# NLTK: stopwords + tokenizer + stemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP_ES = set(stopwords.words("spanish"))
WPT = WordPunctTokenizer()
STEM = SnowballStemmer("spanish")

def canon_label(lbl: str) -> str:
    if lbl is None: return ""
    s = str(lbl).strip().lower().replace(" ", "")
    mapping = {"ods1":"ODS1","od1":"ODS1","1":"ODS1",
               "ods3":"ODS3","od3":"ODS3","3":"ODS3",
               "ods4":"ODS4","od4":"ODS4","4":"ODS4"}
    return mapping.get(s, str(lbl).upper())

def normalize_text(doc: str) -> str:
    if not isinstance(doc, str):
        doc = "" if doc is None else str(doc)
    doc = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]", " ", doc, flags=re.I | re.A)
    doc = doc.lower().strip()
    tokens = WPT.tokenize(doc)
    filtered = [STEM.stem(tok) for tok in tokens if tok not in STOP_ES]
    return " ".join(filtered)

class ColumnMapper(BaseEstimator, TransformerMixin):
    """Acepta 'textos' o 'Textos_espanol' y devuelve Serie de texto."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "textos" in df.columns:
            col = "textos"
        elif "Textos_espanol" in df.columns:
            col = "Textos_espanol"
        else:
            raise ValueError("Falta columna de texto: 'textos' o 'Textos_espanol'.")
        return df[col].fillna("").astype(str)

class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([normalize_text(t) for t in X])

def compute_metrics(model, validation_data_path="Datos_etapa 2.xlsx") -> Dict[str, float]:
    df = pd.read_excel(validation_data_path)
    if "labels" in df.columns:
        y_true = df["labels"].apply(canon_label)
    elif "label" in df.columns:
        y_true = df["label"].apply(canon_label)
    else:
        raise ValueError("El archivo de validación debe tener columna 'labels' o 'label'.")
    y_pred = model.predict(df["textos"])
    
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": float(pm), "recall": float(rm), "f1_score": float(fm)}

# Crear nuevo pipeline SVM con TF-IDF sin entrenamiento
def create_pipeline_svm(tfidf_params: Optional[Dict] = None) -> Pipeline:
    tfidf_defaults = dict(
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    if tfidf_params: tfidf_defaults.update(tfidf_params)
    return Pipeline([
        ("map", ColumnMapper()),
        ("normalize", TextNormalizer()),
        ("tfidf", TfidfVectorizer(**tfidf_defaults)),
        ("svm", LinearSVC())
    ])

def train_svm_with_grid(df: pd.DataFrame, text_col="textos", label_col="labels"):
    X = df[[text_col]].copy()
    y = df[label_col].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    base = create_pipeline_svm()
    param_grid_svm = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__loss": ["hinge", "squared_hinge"],
        "svm__class_weight": [None, "balanced"],
        "tfidf__min_df": [2, 3],
        "tfidf__max_df": [0.85, 0.9],
        "tfidf__ngram_range": [(1,1),(1,2)]
    }
    grid = GridSearchCV(base, param_grid_svm, cv=5, n_jobs=-1, verbose=2, scoring="f1_macro")
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    y_pred = best.predict(X_te)
    print("Mejores params:", grid.best_params_)
    print("F1 macro (CV):", round(grid.best_score_, 4))
    print("\n=== Reporte hold-out ===")
    print(classification_report(y_te, y_pred, digits=3))
    return best, compute_metrics(y_te, y_pred)

class SimpleEnsemble:
    """Votación mayoritaria entre modelo histórico y uno nuevo."""
    def __init__(self, old_model: Pipeline, new_model: Pipeline):
        self.old = old_model; self.new = new_model
        self.models = [m for m in [old_model, new_model] if m is not None]
    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        preds = np.vstack(preds)
        out = []
        for i in range(preds.shape[1]):
            col = preds[:, i]
            vals, cnts = np.unique(col, return_counts=True)
            out.append(vals[np.argmax(cnts)])
        return np.array(out)
    @property
    def steps(self): return [("ensemble", self)]

# Estrategia 1: reentrenamiento desde cero solo con nuevos datos
def full_retrain(current_model, X_new, y_new):
    fresh = create_pipeline_svm()
    X_df = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame({"textos": X_new})
    y_vec = [canon_label(v) for v in y_new]
    fresh.fit(X_df, y_vec)
    y_pred = fresh.predict(X_df)
    return fresh, compute_metrics(fresh)

# Estrategia 2: votación entre modelo histórico y uno nuevo
def ensemble_retrain(current_model, X_new, y_new):
    fresh, _ = full_retrain(None, X_new, y_new)
    if current_model is None:
        X_df = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame({"textos": X_new})
        y_pred = fresh.predict(X_df)
        return fresh, compute_metrics([canon_label(v) for v in y_new], y_pred)
    ens = SimpleEnsemble(current_model, fresh)
    X_df = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame({"textos": X_new})
    y_pred = ens.predict(X_df)
    return ens, compute_metrics(ens)

# Estrategia 3: reentrenamiento desde cero con datos históricos + nuevos
def cumulative_retrain(current_model, X_new, y_new, history_path="docs/dataset_balanceado_etapa2.csv"):
    """
    Reentrena un modelo combinando datos históricos almacenados y nuevos datos.
    """
    # Asegurar estructura
    X_df_new = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame({"textos": X_new})
    y_df_new = pd.Series([canon_label(v) for v in y_new], name="labels")

    # Cargar históricos si existen
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        print(f"Cargados {len(hist)} ejemplos históricos desde {history_path}")
        X_df_hist = hist[["textos"]]
        y_df_hist = hist["labels"]
        # Combinar
        X_all = pd.concat([X_df_hist, X_df_new], ignore_index=True)
        y_all = pd.concat([y_df_hist, y_df_new], ignore_index=True)
    else:
        print("No se encontró dataset histórico, entrenando solo con los datos nuevos.")
        X_all, y_all = X_df_new, y_df_new

    X_all = X_all if isinstance(X_new, pd.DataFrame) else pd.DataFrame({"textos": X_new})
    y_all = pd.Series([canon_label(v) for v in y_all], name="labels")
    #revisar df, y labels
    print(f"Total de datos para reentrenar: {len(X_all)} ejemplos.")
    print("Distribución de clases:\n", y_all.value_counts())

    # Entrenar nuevo modelo con todo
    fresh = create_pipeline_svm()
    fresh.fit(X_all, y_all)
    y_pred = fresh.predict(X_all)
    return fresh, compute_metrics(fresh)


retrain_strategies = {
    "full": full_retrain,
    "ensemble": ensemble_retrain,
    "cumulative": cumulative_retrain
}

def save_model(model: Pipeline, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    from models.model_pipeline import ColumnMapper, TextNormalizer, SimpleEnsemble  # noqa
    return joblib.load(path)


