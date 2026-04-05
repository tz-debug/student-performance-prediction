# Student Academic Risk Prediction and Early Intervention Dashboard

An interactive machine learning dashboard that predicts student academic risk using demographic, behavioral, and academic features. It supports both practical decision support and academically grounded model comparison.

## Files

- `student_academic_risk_dashboard.py`
- `student_performance_updated_1000.csv`
- `requirements.txt`
- `README.md`

## Run

```bash
pip install -r requirements.txt
streamlit run student_academic_risk_dashboard.py
```

## Notes

- This version safely skips models that fail during cross-validation instead of crashing.
- XGBoost is label-encoded internally to support multiclass classification.
- Manual and batch predictions both support XGBoost as well as the other models.
