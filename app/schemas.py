from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Etiquetas en texto; si usas números, ajusta en el model_manager el mapeo.
ODSLabel = Literal["ODS1", "ODS3", "ODS4"]

# Respetar el CSV original: p.ej. "Textos_espanol"
# Permitimos alias "textos" por robustez, pero la doc usa "Textos_espanol".
class Instance(BaseModel):
    Textos_espanol: Optional[str] = Field(None, description="Texto (nombre exacto CSV)")
    textos: Optional[str] = Field(None, description="Alias aceptado (interno)")

class PredictRequest(BaseModel):
    instances: List[Instance] = Field(..., description="Lista de instancias con la columna del CSV")

class PredictResponse(BaseModel):
    predictions: List[str]

class LabeledInstance(Instance):
    label: ODSLabel  # variable objetivo (ODS1/ODS3/ODS4)

class RetrainRequest(BaseModel):
    instances: List[LabeledInstance]
    validation_ratio: Optional[float] = 0.2
    random_state: Optional[int] = 42

class RetrainMetrics(BaseModel):
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict

class RetrainResponse(BaseModel):
    metrics: RetrainMetrics
    saved_as: str
