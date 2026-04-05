import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


st.set_page_config(page_title="Student Academic Risk Dashboard", layout="wide")
st.title("Student Academic Risk Prediction and Early Intervention Dashboard")
st.write(
    "An interactive machine learning dashboard for identifying student academic risk, "
    "comparing classification models, and generating intervention-oriented insights."
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


def get_available_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            class_weight="balanced",
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=6,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost Classifier"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

    return models


def build_model(model_name: str):
    return get_available_models()[model_name]


def summarize_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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


def evaluate_models(X_train, y_train, cv_folds):
    results = []
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    for model_name, model in get_available_models().items():
        preprocessor, _, _ = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=None,
        )

        results.append(
            {
                "Model": model_name,
                "CV Accuracy Mean": np.mean(scores["test_accuracy"]),
                "CV Accuracy Std": np.std(scores["test_accuracy"]),
                "CV Precision Mean": np.mean(scores["test_precision_macro"]),
                "CV Recall Mean": np.mean(scores["test_recall_macro"]),
                "CV F1 Mean": np.mean(scores["test_f1_macro"]),
            }
        )

    return pd.DataFrame(results).sort_values("CV F1 Mean", ascending=False)


def to_numeric_safe(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def intervention_suggestions(row, prediction):
    suggestions = []

    attendance = to_numeric_safe(row.get("AttendanceRate", np.nan))
    study_hours = to_numeric_safe(row.get("StudyHoursPerWeek", np.nan))
    previous_grade = to_numeric_safe(row.get("PreviousGrade", np.nan))
    online_classes = to_numeric_safe(row.get("Online Classes Taken", np.nan))
    support = str(row.get("ParentalSupport", "")).strip().lower()

    if prediction == "Low":
        suggestions.append("Student may require immediate academic support and closer progress monitoring.")

    if pd.notna(attendance) and attendance < 75:
        suggestions.append("Low attendance detected. Consider attendance counselling or academic advising.")
    elif pd.notna(attendance) and attendance < 85:
        suggestions.append("Attendance is moderate. Monitor consistency and reinforce participation.")

    if pd.notna(study_hours) and study_hours < 8:
        suggestions.append("Study time appears low. Recommend structured study planning and time management support.")
    elif pd.notna(study_hours) and study_hours < 12:
        suggestions.append("Study hours are moderate. Additional guided revision may improve outcomes.")

    if pd.notna(previous_grade) and previous_grade < 70:
        suggestions.append("Previous academic performance indicates risk. Consider remedial tutoring or revision sessions.")

    if support in ["low", "none", "weak"]:
        suggestions.append("Limited support context detected. Mentoring or institutional support may be beneficial.")

    if pd.notna(online_classes) and online_classes < 5:
        suggestions.append("Low engagement with online classes. Encourage greater use of digital learning resources.")

    if not suggestions:
        suggestions.append("Current profile indicates stable academic outlook. Maintain present study and attendance habits.")

    return suggestions


def create_batch_prediction_output(input_df, predictions):
    output = input_df.copy()
    output["PredictedPerformance"] = predictions
    output["RiskFlag"] = output["PredictedPerformance"].apply(
        lambda x: "At Risk" if x == "Low" else ("Monitor" if x == "Medium" else "Stable")
    )
    return output


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Configuration")

dataset_option = st.sidebar.selectbox(
    "Choose dataset source",
    ["Built-in Student Dataset", "Upload Your Own"],
)

selected_model = st.sidebar.selectbox(
    "Model for final training",
    list(get_available_models().keys()),
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42)
cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5, 1)
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data", "Model Comparison", "Evaluation", "Single Prediction", "Batch Prediction"]
)


# -------------------------------------------------
# DATA TAB
# -------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    if loaded_path:
        st.caption(f"Loaded from: {loaded_path}")

    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Features", f"{X.shape[1]}")
    c3.metric("Classes", f"{y.nunique()}")

    with st.expander("Dataset information"):
        st.write("Columns:", df.columns.tolist())
        st.write("Numeric features:", numeric_features)
        st.write("Categorical features:", categorical_features)
        st.write("Target distribution:")
        st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"))
        st.write("Available models:", list(get_available_models().keys()))

    st.subheader("Academic Framing")
    st.write(
        "This dashboard can be used both as a practical early warning system for identifying "
        "students at risk and as an academic machine learning study comparing interpretable "
        "and non-linear classification models for education analytics."
    )

    st.subheader("Target Definition")
    st.write("The target variable `Performance` is derived from `FinalGrade` as follows:")
    st.write("- Low: FinalGrade < 70")
    st.write("- Medium: 70 ≤ FinalGrade < 85")
    st.write("- High: FinalGrade ≥ 85")

    if not XGBOOST_AVAILABLE:
        st.warning(
            "XGBoost is not currently available in this environment. "
            "Install it through requirements.txt to enable XGBoost model comparison."
        )


# -------------------------------------------------
# MODEL COMPARISON TAB
# -------------------------------------------------
with tab2:
    st.subheader("Comparative Model Evaluation")
    st.write(
        "This section compares multiple classifiers using cross-validation on the training data. "
        "This supports both practical model selection and academically grounded evaluation."
    )

    if st.button("Run Model Comparison"):
        X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        comparison_df = evaluate_models(X_train_cv, y_train_cv, cv_folds)
        st.session_state["comparison_df"] = comparison_df
        st.dataframe(comparison_df, use_container_width=True)

        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
        ax_cmp.bar(comparison_df["Model"], comparison_df["CV F1 Mean"])
        ax_cmp.set_title("Cross-Validated Macro F1 Score by Model")
        ax_cmp.set_ylabel("Macro F1")
        ax_cmp.tick_params(axis="x", rotation=35)
        st.pyplot(fig_cmp)

        chosen_model = comparison_df.iloc[0]["Model"]
        st.info(f"Best cross-validated model by Macro F1: {chosen_model}")

        model = build_model(selected_model)
        preprocessor_train, _, _ = build_preprocessor(X)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_train),
                ("model", model),
            ]
        )

        pipeline.fit(X_train_cv, y_train_cv)
        y_pred = pipeline.predict(X_test_holdout)

        accuracy = accuracy_score(y_test_holdout, y_pred)
        precision = precision_score(y_test_holdout, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test_holdout, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test_holdout, y_pred, average="macro", zero_division=0)

        st.session_state["pipeline"] = pipeline
        st.session_state["X"] = X
        st.session_state["feature_cols"] = X.columns.tolist()
        st.session_state["metrics"] = {
            "Accuracy": accuracy,
            "Precision_macro": precision,
            "Recall_macro": recall,
            "F1_macro": f1,
        }
        st.session_state["X_test"] = X_test_holdout
        st.session_state["y_test"] = y_test_holdout
        st.session_state["y_pred"] = y_pred
        st.session_state["report_df"] = summarize_report(y_test_holdout, y_pred)
        st.session_state["reference_X"] = X

        st.success(f"Final model '{selected_model}' trained successfully on holdout split.")


# -------------------------------------------------
# EVALUATION TAB
# -------------------------------------------------
with tab3:
    st.subheader("Final Model Evaluation")

    if "pipeline" not in st.session_state:
        st.info("Run Model Comparison first to train the final model.")
    else:
        metrics = st.session_state["metrics"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        pipeline = st.session_state["pipeline"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        m2.metric("Precision (Macro)", f"{metrics['Precision_macro']:.3f}")
        m3.metric("Recall (Macro)", f"{metrics['Recall_macro']:.3f}")
        m4.metric("F1 (Macro)", f"{metrics['F1_macro']:.3f}")

        st.download_button(
            "Download Metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name="academic_risk_metrics.json",
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
        ax_cm.set_title(f"Confusion Matrix - {selected_model}")
        st.pyplot(fig_cm)

        st.subheader("Feature Interpretation")
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
        else:
            st.info("This model does not expose direct feature importance or coefficients in the current setup.")

        st.subheader("Academic and Practical Interpretation")
        st.write(
            "These results can be used in two ways: as an operational tool for flagging "
            "students who may benefit from support, and as a machine learning study of "
            "the factors associated with academic outcomes."
        )

        st.subheader("Ethical and Methodological Notes")
        st.write(
            "- Predictions should support student assistance, not punitive decision-making.\n"
            "- Model outputs depend on the quality and representativeness of the dataset.\n"
            "- Demographic and contextual features should be interpreted carefully to avoid bias."
        )


# -------------------------------------------------
# SINGLE PREDICTION TAB
# -------------------------------------------------
with tab4:
    st.subheader("Single Student Prediction")

    if "pipeline" not in st.session_state:
        st.info("Run Model Comparison first to train the final model.")
    else:
        input_data = {}
        reference_X = st.session_state["reference_X"]

        col1, col2 = st.columns(2)

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
                if options:
                    input_data[feature] = st.selectbox(feature, options=options, index=0)
                else:
                    input_data[feature] = ""

        if st.button("Predict Student Outcome"):
            input_df = pd.DataFrame([input_data])
            pred = st.session_state["pipeline"].predict(input_df)[0]

            if pred == "Low":
                risk_flag = "At Risk"
            elif pred == "Medium":
                risk_flag = "Monitor"
            else:
                risk_flag = "Stable"

            st.success(f"Predicted performance class: {pred}")
            st.info(f"Risk flag: {risk_flag}")

            suggestions = intervention_suggestions(input_df.iloc[0], pred)
            st.subheader("Suggested Academic Actions")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")


# -------------------------------------------------
# BATCH PREDICTION TAB
# -------------------------------------------------
with tab5:
    st.subheader("Batch Prediction for Multiple Students")
    st.write(
        "Upload a CSV containing the same predictor columns used by the model to generate "
        "predictions and risk flags for multiple students at once."
    )

    if "pipeline" not in st.session_state:
        st.info("Run Model Comparison first to train the final model.")
    else:
        batch_file = st.file_uploader("Upload batch prediction CSV", type=["csv"], key="batch_file")

        if batch_file is not None:
            batch_df = load_csv_from_upload(batch_file)
            batch_df = standardize_dataset(batch_df)

            expected_cols = st.session_state["reference_X"].columns.tolist()
            missing_cols = [col for col in expected_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"The uploaded file is missing these required columns: {missing_cols}")
            else:
                batch_input = batch_df[expected_cols].copy()
                batch_preds = st.session_state["pipeline"].predict(batch_input)
                output_df = create_batch_prediction_output(batch_input, batch_preds)

                st.dataframe(output_df.head(25), use_container_width=True)

                st.subheader("Risk Summary")
                st.dataframe(
                    output_df["RiskFlag"].value_counts().rename_axis("RiskFlag").reset_index(name="Count"),
                    use_container_width=True,
                )

                csv_output = output_df.to_csv(index=False)
                st.download_button(
                    "Download Batch Predictions CSV",
                    data=csv_output,
                    file_name="batch_student_risk_predictions.csv",
                    mime="text/csv",
                )
