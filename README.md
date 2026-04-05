# Student Academic Risk Prediction and Early Intervention Dashboard

An interactive machine learning dashboard that predicts student academic risk using demographic, behavioral, and academic features. Built with Streamlit and Scikit-learn, the application combines practical decision support with academically grounded model evaluation.

## Live Demo

Add your Streamlit deployment link here after deployment.

## Overview

This project sits at the intersection of education analytics, applied machine learning, and decision support. It is designed both as a practical early warning system for identifying students who may need intervention and as a structured comparative study of machine learning models in educational settings.

The dashboard predicts student performance categories derived from final grade data and provides model comparison, evaluation metrics, intervention-oriented recommendations, and batch prediction support.

## Key Features

- Compares Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, and XGBoost when available
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
- Includes feature importance and coefficient-based interpretability where available
- Incorporates ethical and methodological notes for responsible use

## Project Structure

```text
student-performance-prediction/
│
├── student_academic_risk_dashboard.py
├── student_performance_updated_1000.csv
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run student_academic_risk_dashboard.py
```

## Purpose

This project demonstrates how machine learning can be used both as a practical analytics tool and as an academically meaningful study. It combines predictive modeling, interpretability, and intervention-oriented insights in a way that is relevant to real-world education systems as well as research-focused applications.
