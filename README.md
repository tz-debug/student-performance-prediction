# Student Academic Risk Prediction and Early Intervention Dashboard

An interactive machine learning dashboard that predicts student academic risk using demographic, behavioral, and academic features. Built with Streamlit and Scikit-learn, the application combines practical decision support with academically grounded model evaluation.

---

## Live Demo

Access the deployed application here:  
https://student-performance-dashboard-2.streamlit.app/

---

## Overview

This project sits at the intersection of education analytics, applied machine learning, and decision support. It is designed both as a practical early warning system for identifying students who may need intervention and as a structured comparative study of machine learning models in educational settings.

The dashboard predicts student performance categories derived from final grade data and provides model comparison, evaluation metrics, intervention-oriented recommendations, and batch prediction support.

---

## Key Features

- Compares multiple models:
  - Logistic Regression  
  - Random Forest  
  - Decision Tree  
  - Gradient Boosting  
  - Support Vector Machine  
  - K-Nearest Neighbors  
  - XGBoost (if installed)

- Cross-validation based model comparison  
- Automated preprocessing:
  - Missing value imputation  
  - Standard scaling for numeric variables  
  - One-hot encoding for categorical variables  

- Evaluation outputs:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Classification report  
  - Confusion matrix  

- Feature interpretation (importance / coefficients where applicable)  
- Manual prediction with intervention suggestions  
- Batch prediction with downloadable results  
- Robust error handling (models that fail are skipped instead of crashing)

---

## Dataset

The dataset includes academic, behavioral, and contextual variables such as:

- Study hours  
- Attendance  
- Previous grade  
- Online classes taken  
- Parental support  
- Extracurricular activity  

### Target Definition

The target variable `Performance` is derived from `FinalGrade`:

- Low: FinalGrade < 70  
- Medium: 70 ≤ FinalGrade < 85  
- High: FinalGrade ≥ 85  

---

## Project Structure

```text
student-performance-prediction/
│
├── student_academic_risk_dashboard.py
├── student_performance_updated_1000.csv
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run student_academic_risk_dashboard.py
```

---

## Workflow

1. Load dataset (built-in or upload CSV)  
2. Explore dataset and feature types  
3. Run model comparison (cross-validation)  
4. Train selected final model  
5. Evaluate performance  
6. Generate predictions:
   - Single student  
   - Batch predictions  

---

## Academic Value

This project demonstrates:

- Comparative machine learning modeling  
- Cross-validation based evaluation  
- Interpretable modeling (feature importance / coefficients)  
- Applied learning analytics  
- Responsible AI considerations  

---

## Industry Relevance

This system can be applied to:

- edtech platforms  
- student retention systems  
- academic monitoring dashboards  
- early intervention programs  

---

## Key Achievements

- Built an end-to-end ML pipeline from preprocessing to deployment  
- Integrated multiple models into a single comparison framework  
- Added intervention-oriented outputs for practical usability  
- Deployed a fully interactive dashboard using Streamlit  

---

## Limitations

- Performance depends on dataset quality  
- Limited hyperparameter tuning  
- No model persistence implemented  
- Fairness analysis can be extended further  

---

## Future Improvements

- Hyperparameter optimization  
- SHAP-based explainability  
- Fairness and bias analysis  
- Model saving/loading  
- Additional datasets and generalization testing  

---

## Technology Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- XGBoost  

---

## Purpose

This project demonstrates how machine learning can be applied both as a practical analytics tool and as an academically meaningful study. It combines predictive modeling, interpretability, and intervention-oriented insights in a way that is relevant to real-world education systems and research applications.
