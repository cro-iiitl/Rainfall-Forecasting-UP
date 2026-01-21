
BASE_DIR = "/content/drive/MyDrive/UP_Rainfall_Prediction_Project"

RAW_DATA_DIR = f"{BASE_DIR}/data/raw"
CLEAN_DATA_DIR = f"{BASE_DIR}/data/cleaned"
FEATURE_DATA_DIR = f"{BASE_DIR}/data/features"

MODEL_DIR = f"{BASE_DIR}/models/trained_models"
SCALER_DIR = f"{BASE_DIR}/models/scalers"

METRICS_DIR = f"{BASE_DIR}/results/metrics"
PLOTS_DIR = f"{BASE_DIR}/results/plots"

LOG_DIR = f"{BASE_DIR}/logs"
REPORT_DIR = f"{BASE_DIR}/reports"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
TARGET_COLUMN = "rainfall_mm"
