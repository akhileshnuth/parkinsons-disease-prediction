import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu  # still imported but not strictly needed

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------------------------------------------------
# Constants (GitHub & Portfolio links)
# -------------------------------------------------------------------------
GITHUB_URL = "https://github.com/akhileshnuth/parkinsons-disease-prediction"  # change if needed
PORTFOLIO_URL = "https://AkhileshNuth-portfolio-app.vercel.app/"  # 🔗 your portfolio (update if needed)

# -------------------------------------------------------------------------
# Modern UI styling
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #0b1120 100%);
            color: #e5e7eb;
        }
        .stApp {
            background-color: transparent;
        }

        /* Reduce top padding & make content tighter */
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 1.2rem !important;
            max-width: 1200px;
        }

        /* SECTION HEADINGS */
        .section-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 0.15rem;
            color: #111827;   /* dark slate */
        }
        .section-subtitle {
            font-size: 0.8rem;
            color: #4b5563;   /* medium gray */
            margin-bottom: 0.4rem;
        }

        /* Result cards */
        .result-box {
            padding: 1.1rem 1rem;
            border-radius: 1rem;
            border: 1px solid rgba(16, 185, 129, 0.5);
            background: rgba(6, 95, 70, 0.4);
        }
        .result-box-ok {
            border: 1px solid rgba(59, 130, 246, 0.6);
            background: rgba(30, 64, 175, 0.45);
        }

        /* Text inputs – compact & consistent */
        .stTextInput > div > input {
            background-color: #f9fafb;
            border-radius: 0.7rem;
            border: 1px solid rgba(148, 163, 184, 0.9);
            color: #111827;
            padding: 6px 10px !important;
            font-size: 0.85rem !important;
            height: 38px !important;
        }
        .stTextInput > div > input:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 1px #38bdf8 !important;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 999px;
            padding: 0.45rem 1.2rem;
            border: none;
            background: linear-gradient(90deg, #22c55e, #22d3ee);
            color: #0f172a;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .stButton>button:hover {
            filter: brightness(1.05);
        }

        /* Small divider */
        .soft-hr {
            border: none;
            border-top: 1px solid rgba(148, 163, 184, 0.35);
            margin: 0.4rem 0 0.3rem 0;
        }

        /* GitHub/Portfolio button style */
        .github-btn {
            display: inline-block;
            padding: 0.5rem 1.4rem;
            margin-top: 0.5rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #22d3ee);
            text-decoration: none;
            color: #0f172a;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .github-btn:hover {
            filter: brightness(1.05);
        }

        /* Tighten main title spacing a bit */
        h1 {
            margin-bottom: 0.4rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Load model (and scaler if available)
# -------------------------------------------------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "parkinsons_model.sav")

with open(model_path, "rb") as f:
    saved_obj = pickle.load(f)

if isinstance(saved_obj, dict) and "model" in saved_obj:
    parkinsons_model = saved_obj["model"]
    scaler = saved_obj.get("scaler", None)
else:
    parkinsons_model = saved_obj
    scaler = None

# -------------------------------------------------------------------------
# Load dataset & example samples
# -------------------------------------------------------------------------
csv_path = os.path.join(working_dir, "parkinsons.csv")
df = None
healthy_sample = None
parkinson_sample = None

feature_columns = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ",
    "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

feature_keys = [
    "fo", "fhi", "flo",
    "jit_pct", "jit_abs", "rap",
    "ppq", "ddp",
    "shimmer", "shimmer_db",
    "apq3", "apq5", "apq",
    "dda", "nhr", "hnr",
    "rpde", "dfa", "spread1", "spread2", "d2", "ppe"
]

feature_map = list(zip(feature_columns, feature_keys))

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    try:
        healthy_sample = df[df["status"] == 0].iloc[0]
    except Exception:
        healthy_sample = None

    try:
        parkinson_sample = df[df["status"] == 1].iloc[0]
    except Exception:
        parkinson_sample = None

# -------------------------------------------------------------------------
# Initialize empty session_state values
# -------------------------------------------------------------------------
for key in feature_keys:
    if key not in st.session_state:
        st.session_state[key] = ""


def set_from_sample(sample_row):
    """Fill values from sample."""
    if sample_row is None:
        st.warning("Sample not available.")
        return
    for col_name, key in feature_map:
        st.session_state[key] = f"{float(sample_row[col_name]):.6f}"


def reset_all_fields():
    """Clear all feature inputs."""
    for key in feature_keys:
        st.session_state[key] = ""


# -------------------------------------------------------------------------
# Top Title + Tabs
# -------------------------------------------------------------------------
st.title("Parkinson's Disease Prediction")

tab_pred, tab_about = st.tabs(["🔍 Prediction", "ℹ️ About"])

# -------------------------------------------------------------------------
# Prediction Tab
# -------------------------------------------------------------------------
with tab_pred:
    st.markdown(
        "Use the controls below to **load a sample** or **enter values manually**. "
        "All fields start empty to avoid confusion and ensure clean input."
    )

    # Sample buttons + reset button in one aligned row
    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        if st.button("👤 Load Healthy Sample", use_container_width=True):
            set_from_sample(healthy_sample)

    with sample_col2:
        if st.button("🧪 Load Parkinson Sample", use_container_width=True):
            set_from_sample(parkinson_sample)

    with sample_col3:
        if st.button("🔄 Reset All Fields", use_container_width=True):
            reset_all_fields()

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    # ------------------------ Feature Inputs: 4-column grid -----------------
    st.markdown("### Voice Feature Inputs")

    col1, col2, col3, col4 = st.columns(4)

    predict = False  # will be set in col4

    with col1:
        st.text_input("MDVP:Fo(Hz)", key="fo")
        st.text_input("MDVP:Fhi(Hz)", key="fhi")
        st.text_input("MDVP:Flo(Hz)", key="flo")
        st.text_input("MDVP:Jitter(%)", key="jit_pct")
        st.text_input("MDVP:Jitter(Abs)", key="jit_abs")
        st.text_input("MDVP:RAP", key="rap")

    with col2:
        st.text_input("MDVP:PPQ", key="ppq")
        st.text_input("Jitter:DDP", key="ddp")
        st.text_input("MDVP:Shimmer", key="shimmer")
        st.text_input("MDVP:Shimmer(dB)", key="shimmer_db")
        st.text_input("Shimmer:APQ3", key="apq3")
        st.text_input("Shimmer:APQ5", key="apq5")

    with col3:
        st.text_input("MDVP:APQ", key="apq")
        st.text_input("Shimmer:DDA", key="dda")
        st.text_input("NHR", key="nhr")
        st.text_input("HNR", key="hnr")
        st.text_input("RPDE", key="rpde")
        st.text_input("DFA", key="dfa")

    with col4:
        st.text_input("spread1", key="spread1")
        st.text_input("spread2", key="spread2")
        st.text_input("D2", key="d2")
        st.text_input("PPE", key="ppe")

        # 🔽 Button here: same row as RPDE (5th element of this column)
        predict = st.button("🔍 Predict Parkinson's Status", use_container_width=True)

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    # ------------------------ Result Section (full width at bottom) ---------
    result_container = st.container()

    with result_container:
        if predict:
            values = [st.session_state[k] for k in feature_keys]

            if any(v.strip() == "" for v in values):
                st.error("Please fill all fields or load a sample before predicting.")
            else:
                try:
                    floats = [float(v) for v in values]
                except Exception:
                    st.error("All fields must be numeric. Please check your inputs.")
                else:
                    arr = np.array(floats).reshape(1, -1)

                    if scaler is not None:
                        arr_df = pd.DataFrame(arr, columns=feature_columns)
                        arr_scaled = scaler.transform(arr_df)
                    else:
                        arr_scaled = arr

                    pred = parkinsons_model.predict(arr_scaled)

                    if pred[0] == 1:
                        st.markdown(
                            """
                            <div class="result-box">
                                <h4>🩺 Parkinson's Disease Detected</h4>
                                <p>
                                    The provided voice features match patterns typically
                                    associated with Parkinson's Disease in this model.
                                </p>
                                <p style="font-size: 0.8rem; opacity: 0.9;">
                                    This is a machine learning based estimation only and
                                    should not be treated as a clinical diagnosis.
                                    Please consult a medical professional for confirmation.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="result-box result-box-ok">
                                <h4>✅ No Parkinson's Detected</h4>
                                <p>
                                    The voice pattern does not indicate Parkinson's Disease
                                    according to this model.
                                </p>
                                <p style="font-size: 0.8rem; opacity: 0.9;">
                                    This result is not a substitute for medical advice.
                                    Always consult a healthcare professional if you have concerns.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# -------------------------------------------------------------------------
# About Tab
# -------------------------------------------------------------------------
with tab_about:
    st.subheader("About this App")
    st.markdown(
        """
        This application demonstrates how **machine learning** can be applied to
        **acoustic voice features** to estimate the likelihood of Parkinson's Disease.

        It is designed as an **educational and experimental tool**, not as a clinical
        diagnostic system.
        """
    )

    about_col1, about_col2 = st.columns([2, 1])

    with about_col1:
        st.markdown("### 🔍 What does the model use?")
        st.markdown(
            """
            The model is trained on voice measurements such as:

            - **Frequency-related features**: average, maximum and minimum vocal fundamental frequency  
            - **Jitter features**: tiny variations in frequency from cycle to cycle  
            - **Shimmer features**: variations in amplitude / loudness  
            - **Noise and harmonic measures**: signal-to-noise ratios and other quality indicators  
            - **Nonlinear measures**: complexity and irregularity in the voice signal  

            These features are commonly extracted from sustained vowel recordings
            using speech processing tools and then used to train classification models.
            """
        )

        st.markdown("### ⚙️ How should this app be used?")
        st.markdown(
            """
            - As a **learning resource** for students and developers working with ML in healthcare  
            - To experiment with **different feature values** and see how they affect predictions  
            - As a starting template for building more advanced decision-support tools  

            ⚠️ **Important:** This app is **not** a certified medical device and should **never**
            be used as the sole basis for any health-related decision.
            """
        )

    with about_col2:
        st.markdown("### 🧱 Tech Stack")
        st.write(
            """
            - **Python**  
            - **Streamlit** for the interactive UI  
            - **NumPy & Pandas** for data handling  
            - **Scikit-learn** (or similar) for the ML model  
            """
        )

        st.markdown("### 🔗 Important Links")

        # GitHub button
        st.markdown(
            f"""
            <a class="github-btn" href="{GITHUB_URL}" target="_blank">
                ⭐ View on GitHub
            </a>
            """,
            unsafe_allow_html=True,
        )

        # Portfolio button
        st.markdown(
            f"""
            <a class="github-btn" href="{PORTFOLIO_URL}" target="_blank">
                🚀 Visit My Portfolio
            </a>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "If you're interested in collaborating, contributing, or connecting for "
            "tech-related projects, feel free to reach out. Always open to exciting opportunities! 😊"
        )

    st.warning(
        "Disclaimer: This application is for educational and demonstration purposes only "
        "and is **not** a medical diagnostic tool."
    )
