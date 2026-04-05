# Student Academic Risk Prediction and Early Intervention Dashboard

An interactive machine learning dashboard that predicts student academic risk using demographic, behavioral, and academic features. Built with Streamlit and Scikit-learn, the application combines practical decision support with academically grounded model evaluation.

---

## Live Demo

Add your Streamlit deployment link here after deployment.

---

## Overview

This project sits at the intersection of education analytics, applied machine learning, and decision support. It is designed both as a practical early warning system for identifying students who may need intervention and as a structured comparative study of interpretable machine learning models in educational settings.

The dashboard predicts student performance categories derived from final grade data and provides model comparison, evaluation metrics, intervention-oriented recommendations, and batch prediction support.

---

## Key Features

- Compares Logistic Regression and Random Forest classifiers
- Uses cross-validation for model comparison
- Performs automated preprocessing:
  - Missing value imputation
  - Standard scaling for numeric variables
  - One-hot encoding for categorical variables
- Provides evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Classification report
  - Confusion matrix
- Generates intervention-oriented guidance for individual student predictions
- Supports batch prediction and downloadable outputs
- Includes feature importance and coefficient-based interpretability
- Incorporates ethical and methodological notes for responsible use

---

## Dataset

The dataset includes academic, behavioral, and contextual variables such as:

- Study hours
- Attendance
- Previous grade
- Online classes taken
- Parental support
- Extracurricular activities

### Target Definition

The target variable is derived from `FinalGrade`:

- Low: FinalGrade < 70
- Medium: 70 ≤ FinalGrade < 85
- High: FinalGrade ≥ 85

This framing makes the project suitable for early identification of academically at-risk students.

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

Clone the repository:

```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run student_academic_risk_dashboard.py
```

---

## Workflow

1. Load the built-in dataset or upload your own CSV
2. Review the cleaned data and target distribution
3. Compare models using cross-validation
4. Train the selected final model
5. Inspect performance metrics and feature interpretation
6. Generate predictions for individual students
7. Upload a batch CSV to flag multiple students at once

---

## Academic Value

This project can be framed as a comparative machine learning study in education analytics. It demonstrates:

- interpretable classification modeling
- structured comparative evaluation
- cross-validation for robust assessment
- practical discussion of ethics and bias
- analysis of factors associated with academic performance

It is suitable for portfolios aimed at academic research, learning analytics, educational data science, and applied machine learning.

---

## Industry Relevance

This project also has strong practical relevance for:

- edtech platforms
- learning analytics systems
- institutional performance monitoring
- student retention and support systems
- early intervention dashboards

The combination of predictive modeling, risk flagging, and intervention-oriented outputs makes it useful as a prototype decision-support tool.

---

## Key Achievements

- Built and deployed an end-to-end machine learning dashboard for student academic risk prediction
- Integrated preprocessing, model comparison, evaluation, and prediction into a single interactive application
- Added intervention-oriented outputs to connect predictions with practical decision support
- Designed the project to be relevant to both academic research and industry-facing analytics roles

---

## Limitations

- Results depend on the quality and representativeness of the dataset
- No persistent model saving has been implemented
- Hyperparameter tuning is limited
- Risk predictions should be used to support students, not to penalize them
- Fairness and subgroup bias analysis can be expanded further

---

## Future Improvements

- Add hyperparameter optimization
- Extend to additional classifiers such as XGBoost or SVM
- Include fairness analysis across student subgroups
- Add SHAP or other local explanation methods
- Support model saving and reloading
- Add screenshot examples and deployment badges to the repository

---

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Purpose

This project demonstrates how machine learning can be used both as a practical analytics tool and as an academically meaningful study. It combines predictive modeling, interpretability, and intervention-oriented insights in a way that is relevant to real-world education systems as well as research-focused applications.
