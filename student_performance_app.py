import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("Student Performance Prediction")
st.write(
    "An interactive Streamlit application for predicting student performance "
    "from demographic, behavioral, and academic features."
)


# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_csv_from_upload(file) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_error = None
    for enc in encodings_to_try:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


@st.cache_data
def read_csv_flexible(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


def load_dataset_safe(possible_filenames):
    possible_paths = []

    for filename in possible_filenames:
        possible_paths.extend(
            [
                filename,
                os.path.join(".", filename),
                os.path.join("data", filename),
                os.path.join(".", "data", filename),
                os.path.join(os.getcwd(), filename),
                os.path.join(os.getcwd(), "data", filename),
            ]
        )

    checked_paths = []
    for path in possible_paths:
        checked_paths.append(path)
        if os.path.exists(path):
            return read_csv_flexible(path), path, checked_paths

    return None, None, checked_paths


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    if "Study Hours" in df.columns and "StudyHoursPerWeek" in df.columns:
        df["StudyHoursPerWeek"] = df["StudyHoursPerWeek"].fillna(df["Study Hours"])
        df = df.drop(columns=["Study Hours"])

    if "Attendance (%)" in df.columns and "AttendanceRate" in df.columns:
        df["AttendanceRate"] = df["AttendanceRate"].fillna(df["Attendance (%)"])
        df = df.drop(columns=["Attendance (%)"])

    drop_cols = [col for col in ["StudentID", "Name"] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def grade_to_class(x):
    if pd.isna(x):
        return np.nan
    if x < 70:
        return "Low"
    if x < 85:
        return "Medium"
    return "High"


def build_preprocessor(X: pd.DataFrame):
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
    )


def summarize_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "Class": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-Score": values.get("f1-score"),
                    "Support": values.get("support"),
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Configuration")

dataset_option = st.sidebar.selectbox(
    "Choose dataset source",
    ["Built-in Student Dataset", "Upload Your Own"],
)

model_name = st.sidebar.selectbox(
    "Select model",
    ["Logistic Regression", "Random Forest Classifier"],
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42)
show_debug = st.sidebar.checkbox("Show file debug info", value=False)


# -------------------------------------------------
# DATASET LOADING
# -------------------------------------------------
uploaded_file = None
df = None
loaded_path = None
checked_paths = []

if dataset_option == "Built-in Student Dataset":
    df, loaded_path, checked_paths = load_dataset_safe(
        ["student_performance_updated_1000.csv"]
    )
    if df is None:
        st.warning("Built-in student_performance_updated_1000.csv was not found. Upload it below.")
        uploaded_file = st.file_uploader(
            "Upload student_performance_updated_1000.csv",
            type=["csv"],
            key="student_upload",
        )
        if uploaded_file is not None:
            df = load_csv_from_upload(uploaded_file)
            loaded_path = "uploaded manually"
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="custom_upload")
    if uploaded_file is not None:
        df = load_csv_from_upload(uploaded_file)
        loaded_path = "uploaded manually"

if show_debug:
    st.subheader("Debug Information")
    try:
        st.write("Current working directory:", os.getcwd())
        st.write("Visible files in current directory:", os.listdir())
        data_dir = os.path.join(os.getcwd(), "data")
        if os.path.exists(data_dir):
            st.write("Visible files in /data:", os.listdir(data_dir))
        else:
            st.write("/data folder not found")
        st.write("Resolved dataset path:", loaded_path)
        st.write("Checked paths:", checked_paths)
    except Exception as e:
        st.write("Debug listing failed:", e)

if df is None:
    st.info("Select a dataset or upload a CSV file to begin.")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------
df = standardize_dataset(df)

if "FinalGrade" not in df.columns:
    st.error("The dataset must include a 'FinalGrade' column.")
    st.stop()

df["Performance"] = df["FinalGrade"].apply(grade_to_class)
df = df.dropna(subset=["Performance"]).copy()

X = df.drop(columns=["FinalGrade", "Performance"])
y = df["Performance"]

if len(X) < 20:
    st.error("Not enough valid rows available after cleaning.")
    st.stop()

preprocessor, numeric_features, categorical_features = build_preprocessor(X)

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Training", "Evaluation", "Prediction"])


# -------------------------------------------------
# DATA TAB
# -------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    if loaded_path:
        st.caption(f"Loaded from: {loaded_path}")

    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Dataset information"):
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Numeric features:", numeric_features)
        st.write("Categorical features:", categorical_features)
        st.write("Target distribution:")
        st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"))

    st.subheader("Target Definition")
    st.write(
        "The classification target is derived from `FinalGrade` using these bands:"
    )
    st.write("- Low: FinalGrade < 70")
    st.write("- Medium: 70 ≤ FinalGrade < 85")
    st.write("- High: FinalGrade ≥ 85")


# -------------------------------------------------
# TRAINING TAB
# -------------------------------------------------
with tab2:
    st.subheader("Model Training")
    st.write(f"Selected model: **{model_name}**")

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        model = build_model(model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report_df = summarize_report(y_test, y_pred)

        st.session_state["pipeline"] = pipeline
        st.session_state["X"] = X
        st.session_state["feature_cols"] = X.columns.tolist()
        st.session_state["target_col"] = "Performance"
        st.session_state["metrics"] = {"Accuracy": accuracy}
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["report_df"] = report_df
        st.session_state["reference_X"] = X
        st.session_state["class_labels"] = sorted(y.unique())

        st.success("Model trained successfully.")


# -------------------------------------------------
# EVALUATION TAB
# -------------------------------------------------
with tab3:
    st.subheader("Model Evaluation")

    if "pipeline" not in st.session_state:
        st.info("Train the model first.")
    else:
        metrics = st.session_state["metrics"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        pipeline = st.session_state["pipeline"]

        m1 = st.columns(1)[0]
        m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")

        st.download_button(
            "Download Metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name="classification_metrics.json",
            mime="application/json",
        )

        st.subheader("Classification Report")
        st.dataframe(st.session_state["report_df"], use_container_width=True)

        st.subheader("Confusion Matrix")
        labels = pipeline.named_steps["model"].classes_
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig_cm)

        st.subheader("Predictions")
        results_df = st.session_state["X_test"].copy()
        results_df["Actual"] = y_test.values
        results_df["Predicted"] = y_pred
        st.dataframe(results_df.head(25), use_container_width=True)

        csv_results = results_df.to_csv(index=False)
        st.download_button(
            "Download Predictions CSV",
            data=csv_results,
            file_name="student_predictions.csv",
            mime="text/csv",
        )

        st.subheader("Model Interpretation")
        trained_model = pipeline.named_steps["model"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        if hasattr(trained_model, "feature_importances_"):
            imp_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": trained_model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)

            st.dataframe(imp_df.head(20), use_container_width=True)

            fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
            top_imp = imp_df.head(15).iloc[::-1]
            ax_imp.barh(top_imp["Feature"], top_imp["Importance"])
            ax_imp.set_title("Top Feature Importances")
            ax_imp.set_xlabel("Importance")
            st.pyplot(fig_imp)

        elif hasattr(trained_model, "coef_"):
            coef_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Coefficient": np.mean(np.abs(trained_model.coef_), axis=0),
                }
            ).sort_values("Coefficient", ascending=False)

            st.dataframe(coef_df.head(20), use_container_width=True)

            fig_coef, ax_coef = plt.subplots(figsize=(9, 5))
            top_coef = coef_df.head(15).iloc[::-1]
            ax_coef.barh(top_coef["Feature"], top_coef["Coefficient"])
            ax_coef.set_title("Top Average Absolute Coefficients")
            ax_coef.set_xlabel("Coefficient")
            st.pyplot(fig_coef)


# -------------------------------------------------
# PREDICTION TAB
# -------------------------------------------------
with tab4:
    st.subheader("Manual Prediction")

    if "pipeline" not in st.session_state:
        st.info("Train the model first.")
    else:
        input_data = {}
        reference_X = st.session_state["reference_X"]

        col1, col2 = st.columns(2)

        # Fixed order for cleaner UI
        numeric_order = [col for col in reference_X.columns if col in reference_X.select_dtypes(include=["number"]).columns]
        categorical_order = [col for col in reference_X.columns if col not in numeric_order]

        with col1:
            st.markdown("**Numeric Inputs**")
            for feature in numeric_order:
                default_val = float(reference_X[feature].median())
                input_data[feature] = st.number_input(
                    feature,
                    value=default_val,
                    format="%.2f",
                )

        with col2:
            st.markdown("**Categorical Inputs**")
            for feature in categorical_order:
                options = sorted(reference_X[feature].dropna().astype(str).unique().tolist())
                default_option = options[0] if options else ""
                input_data[feature] = st.selectbox(feature, options=options, index=0 if options else None)

        if st.button("Predict Performance"):
            input_df = pd.DataFrame([input_data])
            pred = st.session_state["pipeline"].predict(input_df)[0]
            st.success(f"Predicted performance class: {pred}")
