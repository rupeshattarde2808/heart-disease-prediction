from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardio-Scope | Cardiac Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# ---------------------------- Professional dark theme ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0b1020; --panel:#141e30; --border:#2b3850; --muted:#9aa9bf; --text:#f4f7fc; --red:#fb5964; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 82% 0%,rgba(67,87,153,.16),transparent 35%),var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:rgba(11,16,32,.85); }
[data-testid="stSidebar"] { background:#101827; border-right:1px solid var(--border); }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif!important; color:var(--text)!important; letter-spacing:-.025em; }
p,label { color:#d5ddec; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.brand-icon { display:flex; align-items:center; justify-content:center; width:46px; height:46px; border-radius:14px; color:white; font-size:25px; background:linear-gradient(135deg,#fb5964,#c52c50); }
.brand-name { font:800 23px Manrope,sans-serif; letter-spacing:.04em; color:#f8faff; }
.brand-sub { color:var(--muted); font-size:12px; margin-top:2px; }
.eyebrow { color:#fb8790; font-size:11px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:7px; }
.page-title { font:800 31px Manrope,sans-serif; color:#f7f9ff; letter-spacing:-.04em; }
.page-sub { color:var(--muted); font-size:14px; margin:4px 0 22px; }
.panel { background:linear-gradient(145deg,#172136,#111a2a); border:1px solid var(--border); border-radius:16px; padding:18px 20px; min-height:112px; }
.metric-label { color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.06em; }
.metric-value { color:#f8faff; font:800 25px Manrope,sans-serif; margin-top:8px; line-height:1.2; overflow-wrap:anywhere; }
.metric-note { color:#8b9ab2; font-size:11px; margin-top:7px; }
.model-card { background:linear-gradient(155deg,#172136,#111a2a); border:1px solid var(--border); border-radius:15px; padding:17px; min-height:205px; }
.model-card.selected { border-color:#fb5964; box-shadow:inset 0 0 0 1px rgba(251,89,100,.12); }
.model-name { color:#f7f9ff; font:700 15px Manrope,sans-serif; }
.model-tag { color:#ffabb1; background:#3b202d; border-radius:99px; padding:4px 7px; font-size:9px; font-weight:800; }
.model-score { color:#fff; font:800 28px Manrope,sans-serif; margin-top:22px; }
.muted { color:var(--muted); font-size:12px; }
.track { height:7px; border-radius:99px; background:#2a354a; margin-top:12px; overflow:hidden; }
.fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#fb5964,#ff9b88); }
.result { background:#151f32; border:1px solid var(--border); border-radius:14px; padding:18px; }
.stButton>button,.stFormSubmitButton>button { border-radius:9px; font-weight:700; min-height:42px; }
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] { background:linear-gradient(135deg,#fb5964,#df3d58); color:white; border:0; }
div[data-testid="stMetric"] { background:#151f32; border:1px solid var(--border); border-radius:12px; padding:13px 15px; }
div[data-testid="stMetricLabel"] { color:#aebbd0; }
[data-testid="stTabs"] button { color:#9eacc1; font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#ff7e88; }
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div { background:#151f32; border-color:#334057; border-radius:8px; }
.footer { color:#7888a1; font-size:11px; text-align:center; padding:22px 0 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------- Load trained models ----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_available_models():
    available, errors = {}, {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            try:
                available[name] = load_model(str(path))
            except Exception as exc:
                errors[name] = str(exc)
        else:
            errors[name] = f"Missing {filename}"
    return available, errors

try:
    best_model_file = (MODEL_DIR / "best_model.txt").read_text(encoding="utf-8").strip()
    if best_model_file not in MODEL_FILES.values():
        raise ValueError("Unsupported filename in models/best_model.txt")
    models, model_errors = load_available_models()
    if not models:
        raise FileNotFoundError("No trained models found in the models folder.")
    selected_model_name = next(n for n, f in MODEL_FILES.items() if f == best_model_file)
except Exception as exc:
    st.error(f"Unable to load models: {exc}")
    st.info("Run `python train.py` from your project directory and check the models folder.")
    st.stop()

if "assessment" not in st.session_state:
    st.session_state.assessment = None

# ---------------------------- Sidebar inputs ----------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">♥</div>
      <div><div class="brand-name">CARDIO-SCOPE</div>
      <div class="brand-sub">Machine learning analytics</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### Patient profile")
    st.caption("Enter the available clinical features.")

    with st.form("patient_form", border=False):
        with st.expander("01 · Demographics", expanded=True):
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            dataset = st.selectbox("Dataset source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

        with st.expander("02 · Clinical measurements", expanded=True):
            cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000, value=200)
            thalch = st.number_input("Maximum heart rate achieved", min_value=0, max_value=300, value=150)
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])

        with st.expander("03 · Test results", expanded=False):
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["False", "True"])
            restecg = st.selectbox("Resting ECG result", ["normal", "st-t abnormality", "left ventricular hypertrophy"])
            exang = st.selectbox("Exercise-induced angina?", ["False", "True"])
            slope = st.selectbox("Slope of peak exercise ST segment", ["upsloping", "flat", "downsloping"])
            thal = st.selectbox("Thalassemia result", ["normal", "fixed defect", "reversible defect"])

        submitted = st.form_submit_button("Run assessment →", type="primary", width="stretch")

    st.divider()
    st.caption("Educational prototype · Not for clinical diagnosis")


# ---------------------------- Header ----------------------------
left, right = st.columns([4, 1.2], vertical_alignment="center")
with left:
    st.markdown('<div class="eyebrow">CARDIAC ANALYTICS PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Heart Health Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Explore machine-learning classifications and compare model outputs.</div>', unsafe_allow_html=True)
with right:
    st.markdown(
        '<div style="text-align:right;color:#4dd6b5;font-size:12px;font-weight:700">● MODELS AVAILABLE'
        '<div style="color:#8e9db4;font-size:11px;font-weight:400;margin-top:4px">Local inference environment</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------- Run assessment ----------------------------
if submitted:
    patient = {
        "age": age, "sex": sex.lower(), "dataset": dataset, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs == "True",
        "restecg": restecg, "thalch": thalch, "exang": exang == "True",
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    input_df = pd.DataFrame([patient])
    results, errors = {}, {}

    with st.spinner("Cardio-Scope is evaluating the available models..."):
        for name, estimator in models.items():
            try:
                prediction = int(estimator.predict(input_df)[0])
                probabilities = estimator.predict_proba(input_df)[0]
                classes = list(estimator.classes_)
                positive = float(probabilities[classes.index(1)]) * 100 if 1 in classes else 0.0
                negative = float(probabilities[classes.index(0)]) * 100 if 0 in classes else 0.0
                results[name] = {
                    "prediction": prediction,
                    "positive_score": positive,
                    "negative_score": negative,
                }
            except Exception as exc:
                errors[name] = str(exc)

    if results:
        average_score = float(np.mean([r["positive_score"] for r in results.values()]))
        st.session_state.assessment = {
            "patient": patient,
            "models": results,
            "average_score": average_score,
            "errors": errors,
        }
    else:
        st.session_state.assessment = None
        st.error("No model could complete the prediction. Check the input categories and saved pipelines.")

assessment = st.session_state.assessment
# ---------------------------- Summary cards ----------------------------
cards = st.columns(4)
summary = [
    ("AVAILABLE MODELS", f"{len(models):02d}", "Loaded from models/"),
    ("INPUT FEATURES", "14", "Patient data fields"),
    ("SELECTED MODEL", selected_model_name, "Read from best_model.txt"),
    ("ASSESSMENT STATUS", "Completed" if assessment else "Awaiting input", "Run the form in the sidebar"),
]
for col, (label, value, note) in zip(cards, summary):
    with col:
        size = "19px" if len(value) > 15 else "25px"
        st.markdown(
            f'<div class="panel"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="font-size:{size}">{value}</div>'
            f'<div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
tab_overview, tab_models, tab_about = st.tabs(["Assessment overview", "Model comparison", "About the system"])

with tab_overview:
    st.markdown("### Assessment overview")
    st.markdown('<div class="page-sub">Patient snapshot, class predictions and model score summary.</div>', unsafe_allow_html=True)

    if not assessment:
        a, b = st.columns([1.3, 1])
        with a:
            st.markdown("""
            <div class="panel">
              <div class="eyebrow">GET STARTED</div>
              <div style="font:800 22px Manrope;color:#f7f9ff;margin-bottom:10px">Your assessment workspace</div>
              <div style="color:#aebbd0;line-height:1.75;font-size:14px">
                Enter the 14 model input features in the sidebar and run an assessment.
                Cardio-Scope will compare the available classifiers and visualize their outputs.
              </div>
            </div>""", unsafe_allow_html=True)
        with b:
            st.markdown("""
            <div class="panel">
              <div class="eyebrow">WORKFLOW</div>
              <p><b>01 · Enter patient data</b><br><span style="color:#aebbd0">Provide the available measurements.</span></p>
              <p><b>02 · Run the models</b><br><span style="color:#aebbd0">Generate class predictions and scores.</span></p>
              <p><b>03 · Review the output</b><br><span style="color:#aebbd0">Compare models and explore charts.</span></p>
            </div>""", unsafe_allow_html=True)
    else:
        patient, results = assessment["patient"], assessment["models"]
        pcols = st.columns(4)
        pcols[0].metric("Age", f"{patient['age']} years")
        pcols[1].metric("Sex", patient["sex"].title())
        pcols[2].metric("Blood pressure", f"{patient['trestbps']} mm Hg")
        pcols[3].metric("Cholesterol", f"{patient['chol']} mg/dl")
        selected = results.get(selected_model_name)
        summary_left, summary_right = st.columns([1.2, 1])
        with summary_left:
            if selected:
                label = "Positive class predicted" if selected["prediction"] == 1 else "Negative class predicted"
                color = "#ff9ca4" if selected["prediction"] == 1 else "#8ce5c7"
                desc = ("The selected model classified this input as class 1. This output is not a diagnosis."
                        if selected["prediction"] == 1 else
                        "The selected model classified this input as class 0. This does not rule out heart disease.")
                st.markdown(
                    f'<div class="panel"><div class="eyebrow">SELECTED MODEL RESULT</div>'
                    f'<div style="font:800 20px Manrope;color:#f7f9ff">{label}</div>'
                    f'<div class="muted" style="margin-top:8px">{desc}</div>'
                    f'<div style="margin-top:14px;color:{color};font-size:12px;font-weight:800">{selected_model_name}</div></div>',
                    unsafe_allow_html=True,
                )
        with summary_right:
            st.markdown(
                f'<div class="panel"><div class="eyebrow">AVERAGE POSITIVE-CLASS SCORE</div>'
                f'<div class="metric-value">{assessment["average_score"]:.2f}%</div>'
                f'<div class="metric-note">Simple average across models that completed prediction</div></div>',
                unsafe_allow_html=True,
            )

        st.caption("The average is a dashboard summary, not a separately trained or clinically calibrated ensemble.")
        st.write("")
        st.markdown("### Model output cards")
        model_cols = st.columns(len(results))
        for col, (name, result) in zip(model_cols, results.items()):
            selected_class = name == selected_model_name
            score = min(max(result["positive_score"], 0), 100)
            with col:
                st.markdown(
                    f'<div class="model-card {"selected" if selected_class else ""}">'
                    f'<div style="display:flex;justify-content:space-between;gap:6px;align-items:center">'
                    f'<div class="model-name">{name}</div>'
                    f'{"<span class=\\'model-tag\\'>SELECTED</span>" if selected_class else ""}</div>'
                    f'<div class="model-score">{result["positive_score"]:.2f}%</div>'
                    f'<div class="muted">Positive-class score</div>'
                    f'<div class="track"><div class="fill" style="width:{score:.2f}%"></div></div>'
                    f'<div class="muted" style="margin-top:13px">Predicted class: <b style="color:#f8faff">{result["prediction"]}</b></div>'
                    f'<div class="muted" style="margin-top:5px">Negative-class score: {result["negative_score"]:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.write("")
        c1, c2 = st.columns(2)
        score_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Positive-class score (%)": [results[n]["positive_score"] for n in results],
            "Negative-class score (%)": [results[n]["negative_score"] for n in results],
        })
        with c1:
            st.markdown("#### Class score comparison")
            st.bar_chart(score_df.set_index("Model")[["Negative-class score (%)", "Positive-class score (%)"]], width="stretch")
        with c2:
            st.markdown("#### Positive-class scores")
            st.line_chart(score_df.set_index("Model")[["Positive-class score (%)"]], width="stretch")
            st.caption("The line connects different models; it does not represent change over time.")

        if assessment["errors"]:
            with st.expander("Model prediction warnings"):
                for name, message in assessment["errors"].items():
                    st.warning(f"{name}: {message}")

        st.warning(
            "Educational use only. These class scores are not clinically validated heart-disease "
            "or heart-attack risk percentages. Consult a qualified healthcare professional for medical concerns."
        )
        if st.button("Clear assessment", width="content"):
            st.session_state.assessment = None
            st.rerun()
with tab_models:
    st.markdown("### Model comparison")
    st.markdown('<div class="page-sub">Previously measured held-out test accuracy from your training run.</div>', unsafe_allow_html=True)
    metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Test accuracy (%)": [83.70, 84.78, 85.87],
    })
    st.bar_chart(metrics.set_index("Model"), y="Test accuracy (%)", width="stretch")
    st.dataframe(metrics, width="stretch", hide_index=True)
    st.info(
        "These are previously reported metrics, not recalculated during this app session. "
        "Accuracy alone does not establish clinical suitability; also evaluate precision, recall, "
        "F1-score, confusion matrix, ROC-AUC and calibration."
    )
    st.markdown("#### Loaded model files")
    model_table = pd.DataFrame([
        {"Model": name, "File": filename, "Status": "Loaded" if name in models else "Unavailable"}
        for name, filename in MODEL_FILES.items()
    ])
    st.dataframe(model_table, width="stretch", hide_index=True)


with tab_about:
    st.markdown("### About Cardio-Scope")
    st.write(
        "Cardio-Scope is an educational machine-learning dashboard for exploring binary heart disease "
        "classification using patient health data. It compares outputs from Logistic Regression, "
        "Random Forest and XGBoost."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Input features")
        st.write(
            "Age, sex, dataset source, chest pain type, resting blood pressure, cholesterol, fasting "
            "blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, "
            "ST slope, major vessels and thalassemia category."
        )
    with right:
        st.markdown("#### Technology")
        st.write("Python · Streamlit · Pandas · NumPy · Joblib · Scikit-learn · XGBoost")
    st.markdown("#### Limitations")
    st.write("- Class 0 and class 1 are the binary labels used during training.")
    st.write("- The application does not identify a specific heart disease subtype.")
    st.write("- Model scores are not clinically calibrated risk estimates.")
    st.write("- Do not use this application for diagnosis, treatment or emergency decisions.")

st.markdown(
    '<div class="footer">CARDIO-SCOPE · Educational machine-learning project · Not a medical diagnostic tool</div>',
    unsafe_allow_html=True,
)


