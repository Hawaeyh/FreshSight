from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

EVALUATION_OUTPUTS_DIR = BASE_DIR / "evaluation" / "outputs"
EVALUATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

EXPLAINABILITY_DIR = OUTPUTS_DIR / "explainability"
EXPLAINABILITY_DIR.mkdir(exist_ok=True)

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "ai" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"
TRAINING_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app.db"
