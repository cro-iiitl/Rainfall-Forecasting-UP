---
description: how to run the UP Rainfall Prediction project
---

1. Install dependencies:
// turbo
```powershell
pip install -r requirements.txt
```

2. Sync the model for environment compatibility:
// turbo
```powershell
python scripts/retrain_model.py
```

3. Launch the Streamlit application:
// turbo
```powershell
streamlit run app/frontend/app.py
```
