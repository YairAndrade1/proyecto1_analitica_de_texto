# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd, numpy as np, os, json, joblib
from datetime import datetime
from models.model_pipeline import retrain_strategies
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.getenv("MODEL_PATH", "docs/svm_pipeline_etapa2.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "data/schema.json")

# -------- Helpers --------
def canon_label(lbl):
    if lbl is None: return None
    s = str(lbl).strip().lower().replace(" ", "")
    m = {"ods1":"ODS1","od1":"ODS1","1":"ODS1",
         "ods3":"ODS3","od3":"ODS3","3":"ODS3",
         "ods4":"ODS4","od4":"ODS4","4":"ODS4"}
    return m.get(s, str(lbl).upper())

def load_schema():
    try:
        with open(SCHEMA_PATH, "r") as f: return json.load(f)
    except Exception:
        os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
        default = {"features":["textos"], "target":"labels", "required_columns":["textos","labels"]}
        with open(SCHEMA_PATH, "w") as f: json.dump(default, f, indent=2)
        return default

def coerce_columns(df, schema):
    df = df.copy()
    if "textos" not in df.columns and "Textos_espanol" in df.columns:
        df["textos"] = df["Textos_espanol"]
    for c in ("textos","Textos_espanol"):
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    return df

def parse_retrain_payload(data, schema):
    if "training_data" in data:  # legacy
        df = pd.DataFrame(data["training_data"])
        df = coerce_columns(df, schema)
        if "labels" not in df.columns: raise ValueError("Falta 'labels' en training_data.")
        df["labels"] = df["labels"].apply(canon_label)
        return df[["textos","labels"]], data.get("retrain_strategy","full")
    if "instances" in data:      # canonical
        df = pd.DataFrame(data["instances"])
        df = coerce_columns(df, schema)
        if "label" not in df.columns: raise ValueError("Falta 'label' en instances.")
        df["labels"] = df["label"].apply(canon_label)
        return df[["textos","labels"]], data.get("retrain_strategy","full")
    raise ValueError("Payload inválido: use 'instances' o 'training_data'.")

schema = load_schema()


# Asegura stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except Exception:
        pass  # si falla SSL, igual seguimos con un fallback simple

_spanish_stopwords = set(stopwords.words("spanish")) if 'stopwords' in globals() else {
    "a","ante","bajo","cabe","con","contra","de","desde","en","entre","hacia","hasta","para","por","segun","sin","so","sobre","tras",
    "el","la","los","las","un","una","unos","unas","lo","al","del","y","o","u","ni","que","como","mas","menos","pero","sino","ya","le","les","se",
    "su","sus","mi","mis","tu","tus","nos","vos","ellos","ellas","usted","ustedes","esto","esta","estos","estas","ese","esa","esos","esas","aqui",
    "alli","alla","ser","estar","hacer","poder","haber","tener"
}
_wpt = WordPunctTokenizer()
_stem = SnowballStemmer("spanish")

def _normalize_text(doc: str) -> str:
    if not isinstance(doc, str):
        doc = "" if doc is None else str(doc)
    # Limpieza de caracteres (mantén tildes)
    doc = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', doc, flags=re.I|re.A)
    doc = doc.lower().strip()
    # Tokenización + stopwords + stemming
    tokens = _wpt.tokenize(doc)
    filtered = [_stem.stem(tok) for tok in tokens if tok not in _spanish_stopwords]
    return ' '.join(filtered)

def norm_all_data(data):
    """
    Shim compatible con FunctionTransformer(norm_all_data):
    recibe array/serie de textos y devuelve array normalizado.
    """
    v = np.vectorize(_normalize_text)
    # evita fallos si llega DataFrame['col']/Series/numpy array
    try:
        return v(data.astype(str))
    except Exception:
        return v(np.array(data, dtype=str))

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Importar clases personalizadas antes de cargar, por seguridad
            from models.model_pipeline import ColumnMapper, TextNormalizer
            model = joblib.load(MODEL_PATH)
            print(f"[OK] Modelo cargado: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"[ERR] Cargando modelo: {e}")
            return None
    else:
        print(f"[WARN] Modelo no encontrado en {MODEL_PATH}")
        return None

model = load_model()

# -------- Endpoints --------
@app.get("/health")
def health():
    return jsonify({
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    })

@app.post("/predict")
def predict():
    try:
        if model is None: return jsonify({"error":"Modelo no disponible"}), 500
        data = request.get_json(silent=True) or {}
        instances = data.get("instances", [])
        if not isinstance(instances, list) or not instances:
            return jsonify({"error":"Missing or empty 'instances' list"}), 400
        df = pd.DataFrame(instances)
        df = coerce_columns(df, schema)
        if "textos" not in df.columns: return jsonify({"error":"Missing 'textos'/'Textos_espanol'"}), 400
        X = df["textos"]
        preds = model.predict(X)
        preds = [canon_label(p) for p in preds]
        return jsonify({"predictions": preds, "count": len(preds), "timestamp": datetime.now().isoformat()}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.post("/retrain")
def retrain():
    global model
    try:
        if model is None: return jsonify({"error":"Modelo no disponible"}), 500
        data = request.get_json(silent=True) or {}
        df, strategy = parse_retrain_payload(data, schema)   # -> textos,labels
        if strategy not in retrain_strategies:
            return jsonify({"error": f"Invalid strategy. Available: {list(retrain_strategies.keys())}"}), 400
        X = df[["textos"]]; y = df["labels"]
        trained_model, metrics = retrain_strategies[strategy](model, X, y)
        joblib.dump(trained_model, MODEL_PATH)
        model = trained_model
        return jsonify({
            "message": f"Model retrained using '{strategy}'",
            "metrics": metrics,
            "strategy_used": strategy,
            "training_instances": int(len(df)),
            "saved_as": os.path.basename(MODEL_PATH),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {e}"}), 500
    
@app.get("/model/info")
def model_info():
    if model is None:
        return jsonify({"error": "Modelo no disponible"}), 500
    info = {
        "type": type(model).__name__,
        "loaded_from": MODEL_PATH,
        "steps": [],
        "has_tfidf_idf": None,
        "sklearn_version": __import__("sklearn").__version__,
    }
    if hasattr(model, "steps"):
        info["steps"] = [name for name, _ in model.steps]
        try:
            tfidf = dict(model.steps).get("tfidf")
            info["has_tfidf_idf"] = hasattr(tfidf, "idf_")
        except Exception:
            info["has_tfidf_idf"] = False
    return jsonify(info), 200

@app.post("/model/sanity")
def model_sanity():
    """Prueba una predicción mínima para detectar errores de vectorizador no fitted."""
    try:
        X = pd.Series(["vacunación en el barrio", "educación primaria", "pobreza extrema"])
        _ = model.predict(X)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/expert")
def expert():
    return render_template("expert.html")



if __name__ == "__main__":
    print(f"Modelo cargado: {model is not None} | {MODEL_PATH}")
    app.run(debug=True, host="0.0.0.0", port=8000)
