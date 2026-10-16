import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Cardio-Scope | Heart Disease AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


st.markdown("""
<style>
    .stApp {
        background-color: #101116;
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] {
        background-color: #20212B;
        border-right: 1px solid #30313C;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }

    .main-title {
        font-size: 35px;
        font-weight: 800;
        color: #F5F5F7;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #B7B9C5;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
        color: #F5F5F7;
    }

    .model-card {
        background: #191A22;
        border: 1px solid #343540;
        border-radius: 12px;
        padding: 20px;
        min-height: 235px;
    }

    .model-name {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 8px;
        color: #F5F5F7;
    }

    .model-description {
        font-size: 12px;
        color: #B7B9C5;
        margin-bottom: 16px;
    }

    .score-pill {
        border-radius: 7px;
        padding: 12px;
        font-size: 15px;
        font-weight: 700;
        margin: 12px 0;
    }

    .high {
        background: #51282D;
        color: #FFB4BA;
    }

    .moderate {
        background: #44411F;
        color: #F4E99B;
    }

    .lower {
        background: #193B32;
        color: #A8F0D1;
    }

    .combined-card {
        background: #191A22;
        border: 1px solid #343540;
        border-radius: 12px;
        padding: 20px;
    }

    .big-score {
        font-size: 34px;
        font-weight: 800;
        color: #FFFFFF;
    }

    .small-muted {
        font-size: 12px;
        color: #B7B9C5;
    }

    div.stButton > button[kind="primary"] {
        background-color: #F0444B;
        color: white;
        border: none;
        border-radius: 8px;
        min-height: 44px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #D9323B;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: #191A22;
        border: 1px solid #343540;
        padding: 15px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #B7B9C5;
    }

    hr {
        border-color: #343540;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_resource
def load_all_models():
    loaded = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if model_path.exists():
            loaded[model_name] = load_model(str(model_path))

    return loaded


try:
    with open(MODEL_DIR / "best_model.txt", "r") as file:
        best_model_file = file.read().strip()

    allowed_files = set(MODEL_FILES.values())

    if best_model_file not in allowed_files:
        raise ValueError(
            "Invalid model filename in models/best_model.txt"
        )

    models = load_all_models()

    if not models:
        raise FileNotFoundError(
            "No trained model files were found in the models folder."
        )

    selected_model_name = next(
        name for name, filename in MODEL_FILES.items()
        if filename == best_model_file
    )

except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.info("Run `python train.py` from your project directory.")
    st.stop()


if "assessment" not in st.session_state:
    st.session_state.assessment = None


with st.sidebar:
    st.markdown("## Patient Health Information")
    st.caption("Enter the patient's available health measurements.")
    st.divider()

    with st.form("patient_form"):

        st.markdown("###  Demographics")

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        dataset = st.selectbox(
            "Dataset category",
            ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
        )

        st.markdown("### Clinical Measurements")

        cp = st.selectbox(
            "Chest pain type",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ]
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=0,
            max_value=300,
            value=120
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=0,
            max_value=1000,
            value=200
        )

        thalch = st.number_input(
            "Maximum heart rate achieved",
            min_value=0,
            max_value=300,
            value=150
        )

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        ca = st.selectbox(
            "Number of major vessels (0–3)",
            [0, 1, 2, 3]
        )

        st.markdown("###  Test Results")

        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl?",
            ["False", "True"]
        )

        restecg = st.selectbox(
            "Resting ECG result",
            [
                "normal",
                "st-t abnormality",
                "left ventricular hypertrophy"
            ]
        )

        exang = st.selectbox(
            "Exercise-induced angina?",
            ["False", "True"]
        )

        slope = st.selectbox(
            "Slope of peak exercise ST segment",
            ["upsloping", "flat", "downsloping"]
        )

        thal = st.selectbox(
            "Thalassemia result",
            ["normal", "fixed defect", "reversible defect"]
        )

        submitted = st.form_submit_button(
            " Assess Model Scores",
            type="primary",
            width="stretch"
        )


# --------------------------------------------------
# MAIN DASHBOARD HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title"> Cardio-Scope AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Heart disease classification and machine-learning model comparison'
    '</div>',
    unsafe_allow_html=True
)

tab1,tab2 = st.tabs([" Risk Assessment"," Model Information"])

if submitted:
    patient_data = {
        "age": age,
        "sex": sex.lower(),
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs == "True",
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang == "True",
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([patient_data])

    with st.spinner("Model is predicting... Please wait."):
        try:
            time.sleep(1)

            model_results = {}

            for model_name, current_model in models.items():
                prediction = int(current_model.predict(input_df)[0])

                probabilities = current_model.predict_proba(input_df)[0]
                classes = list(current_model.classes_)

                if 1 in classes:
                    positive_score = float(
                        probabilities[classes.index(1)]
                    ) * 100
                else:
                    positive_score = 0.0

                if 0 in classes:
                    negative_score = float(
                        probabilities[classes.index(0)]
                    ) * 100
                else:
                    negative_score = 0.0

                model_results[model_name] = {
                    "prediction": prediction,
                    "positive_score": positive_score,
                    "negative_score": negative_score
                }

            # Equal-weight average of available model scores.
            # This is a dashboard summary, not a calibrated clinical risk.
            combined_score = float(np.mean([
                result["positive_score"]
                for result in model_results.values()
            ]))

            st.session_state.assessment = {
                "patient": patient_data,
                "models": model_results,
                "combined_score": combined_score
            }

        except Exception as error:
            st.session_state.assessment = None
            st.error(f"Prediction failed: {error}")



with tab1:

    st.markdown(
        '<div class="section-title"> Heart Disease Assessment</div>',
        unsafe_allow_html=True
    )

    if st.session_state.assessment is None:
        st.info(
            "Enter the patient's details in the left sidebar, "
            "then click **Assess Model Scores**."
        )

        st.markdown("### What this dashboard provides")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("####  Model comparison")
            st.write("Compare predictions from available trained models.")

        with c2:
            st.markdown("####  Score visualization")
            st.write("Review positive-class model scores and charts.")

        with c3:
            st.markdown("####  Model selection")
            st.write("See which model was selected during training.")

    else:
        assessment = st.session_state.assessment
        patient = assessment["patient"]
        results = assessment["models"]
        combined_score = assessment["combined_score"]

        # Patient summary
        st.markdown("###  Patient Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Age", f"{patient['age']} years")

        with c2:
            st.metric("Sex", patient["sex"].title())

        with c3:
            st.metric("Blood Pressure", f"{patient['trestbps']} mm Hg")

        with c4:
            st.metric("Cholesterol", f"{patient['chol']} mg/dl")

        st.divider()

        st.markdown("### Risk Assessment Results")

        model_names = list(results.keys())
        columns = st.columns(len(model_names))

        for column, model_name in zip(columns, model_names):
            result = results[model_name]
            score = result["positive_score"]

            # Visual score categories only; not clinical thresholds.
            if score >= 70:
                category = "Higher positive-class score"
                css_class = "high"
            elif score >= 40:
                category = "Intermediate positive-class score"
                css_class = "moderate"
            else:
                category = "Lower positive-class score"
                css_class = "lower"

            with column:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-name"> {model_name}</div>
                        <div class="model-description">
                            {"Selected during training"
                             if MODEL_FILES[model_name] == best_model_file
                             else "Additional model comparison"}
                        </div>
                        <div class="score-pill {css_class}">
                            {category}: {score:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(score / 100, 0.0), 1.0)
                )

                st.caption(
                    f"Predicted class: {result['prediction']}"
                )

                st.caption(
                    f"Negative score: {result['negative_score']:.2f}%"
                )

        st.caption(
            "The categories above describe model scores for comparison. "
            "They are not medical risk categories or validated "
            "heart-attack probabilities."
        )

        st.divider()

        # Combined assessment
        st.markdown("###  Combined Model Summary")

        left, right = st.columns([1.3, 1])

        with left:
            if combined_score >= 70:
                summary_label = "Higher positive-class score"
            elif combined_score >= 40:
                summary_label = "Intermediate positive-class score"
            else:
                summary_label = "Lower positive-class score"

            st.markdown(
                f"""
                <div class="combined-card">
                    <div class="small-muted">
                        Average positive-class score across available models
                    </div>
                    <div class="big-score">{combined_score:.2f}%</div>
                    <div>{summary_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown("#### Model summary")

            chart_df = pd.DataFrame({
                "Model": model_names,
                "Positive-class score (%)": [
                    results[name]["positive_score"]
                    for name in model_names
                ]
            })

            st.bar_chart(
                chart_df.set_index("Model"),
                y="Positive-class score (%)"
            )

        st.markdown("###  Model Score Trend")

        line_df = chart_df.set_index("Model")
        st.line_chart(line_df)

        st.caption(
            "The line connects model scores for visual comparison; "
            "it does not represent risk changing over time."
        )

        st.divider()

        st.markdown("###  Prediction Details")

        st.write(f"**Selected training model:** {selected_model_name}")

        selected_result = results.get(selected_model_name)

        if selected_result:
            st.write(
                f"**Selected model's predicted class:** "
                f"{selected_result['prediction']}"
            )

        st.warning(
            "Educational project only. These models cannot diagnose "
            "heart disease or estimate a clinically validated "
            "heart-attack risk. Consult a qualified healthcare "
            "professional for medical concerns."
        )

        if st.button("🔄 Clear Results", width="stretch"):
            st.session_state.assessment = None
            st.rerun()


with tab2:

    st.markdown("###  Model Information")

    st.write(
        "This application loads the trained models saved in the "
        "`models` directory."
    )

    st.markdown(f"**Model selected during training:** `{selected_model_name}`")

    model_info = pd.DataFrame([
        {
            "Model": name,
            "File": filename,
            "Status": "Loaded" if name in models else "File not found"
        }
        for name, filename in MODEL_FILES.items()
    ])

    st.dataframe(
        model_info,
        width="stretch",
        hide_index=True
    )

    st.markdown("#### Understanding the output")

    st.write(
        "- **Predicted class 0:** the model predicts the negative class."
    )
    st.write(
        "- **Predicted class 1:** the model predicts the positive class."
    )
    st.write(
        "- **Positive-class score:** the model's estimated probability "
        "for class 1, not a medically validated risk percentage."
    )
    st.write(
        "- **Combined score:** the simple average of the available "
        "models' positive-class scores. It is not a separately trained "
        "or clinically calibrated ensemble."
    )

    st.info(
        "Model comparison on a single patient does not establish which "
        "model is most accurate. Evaluate accuracy, precision, recall, "
        "F1-score and other metrics on held-out data."
    )