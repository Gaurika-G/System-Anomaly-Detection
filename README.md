# Network Traffic Anomaly Detection System

## Phase 1 — Data Pipeline

### Project Structure

```
anomaly_detection/
├── config/
│   └── config.yaml           # All settings in one place
├── data/
│   ├── UNSW_NB15_training-set.csv   # ← place dataset here
│   └── UNSW_NB15_testing-set.csv    # ← place dataset here
├── src/
│   ├── __init__.py
│   ├── logger.py             # Centralized logging
│   ├── ingestion.py          # Load & validate raw data
│   ├── preprocessing.py      # Clean, encode, scale
│   └── feature_engineering.py  # Create new features
├── outputs/                  # Auto-created on first run
├── logs/                     # Auto-created on first run
├── pipeline.py               # Run the full pipeline
└── requirements.txt
```

### Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the two CSV files in the `data/` folder:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

### Run the Pipeline

```bash
cd anomaly_detection
python pipeline.py
```

### Output

After running, you'll find:
- `outputs/processed_train.parquet` — cleaned, encoded, scaled training data
- `outputs/processed_test.parquet`  — cleaned, encoded, scaled test data
- `outputs/encoders.pkl`            — saved encoders & scaler for later use
- `logs/pipeline.log`               — full run log with timestamps
