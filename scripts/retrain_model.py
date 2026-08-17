import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "."
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "phase3_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "phase7_rain_classifier")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading features from {FEATURES_PATH}...")
# -----------------------------
# Load features
# -----------------------------
df = pd.read_csv(FEATURES_PATH)

# Recreate target (same Phase 7 rule)
df["rain_binary"] = (df["rain_t_plus_1"] > 0).astype(int)

# -----------------------------
# Train / Val / Test split (same logic as notebook)
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["rain_binary"],
    random_state=42
)

# -----------------------------
# Features & target
# -----------------------------
X_train = train_df.drop(columns=["rain_binary"])
y_train = train_df["rain_binary"]

# -----------------------------
# Preprocessing
# -----------------------------
categorical_cols = ["district"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

print(f"Features: {len(X_train.columns)} columns")
print(f"Categorical: {categorical_cols}")
print(f"Numeric: {len(numeric_cols)} columns")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# -----------------------------
# Final Classifier (Phase 7 best)
# -----------------------------
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", classifier)
])

# -----------------------------
# Train FINAL model
# -----------------------------
print("Training model... (This may take a moment)")
pipeline.fit(X_train, y_train)

# -----------------------------
# Save artifacts
# -----------------------------
print(f"Saving artifacts to {MODEL_DIR}...")
joblib.dump(pipeline, os.path.join(MODEL_DIR, "rain_no_rain_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "classifier_preprocessor.pkl"))

THRESHOLD = 0.40
with open(os.path.join(MODEL_DIR, "threshold.txt"), "w") as f:
    f.write(str(THRESHOLD))

print("✅ FINAL Rain / No-Rain model saved successfully")
print("📂 Model directory:", MODEL_DIR)
print("📌 Threshold:", THRESHOLD)
