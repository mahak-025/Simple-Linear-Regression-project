import streamlit as st
import numpy as np
import pickle
import time

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        }

        /* Title styling */
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(90deg, #f7971e, #ffd200);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            padding-top: 10px;
        }

        .sub-title {
            text-align: center;
            color: #d1d9e6;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }

        /* Card container */
        .prediction-card {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-top: 20px;
        }

        /* Result box */
        .result-box {
            background: linear-gradient(90deg, #11998e, #38ef7d);
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            margin-top: 25px;
            animation: fadeIn 0.6s ease-in-out;
        }

        .result-text {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
        }

        .result-label {
            font-size: 1rem;
            color: #e0f7f0;
            margin-bottom: 5px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Slider label */
        label, .stSlider label {
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Predict button */
        div.stButton > button {
            background: linear-gradient(90deg, #f7971e, #ffd200);
            color: #1a1a1a;
            font-weight: 700;
            font-size: 1.1rem;
            border-radius: 12px;
            padding: 10px 0px;
            width: 100%;
            border: none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 210, 0, 0.4);
            color: #000000;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #8ea0b5;
            font-size: 0.85rem;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="main-title">💰 Salary Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Estimate expected salary based on years of experience</p>', unsafe_allow_html=True)

# ----------------------------
# Input Card
# ----------------------------
st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

years_experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=3.0,
    step=0.5,
    help="Drag to set number of years of professional experience"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔮 Predict Salary")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_clicked:
    with st.spinner("Calculating prediction..."):
        time.sleep(0.6)  # small delay purely for a smoother feel

        prediction = model.predict(np.array([[years_experience]]))[0]

        st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Annual Salary</div>
                <div class="result-text">₹ {prediction:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown('<p class="footer">Built with ❤️ using Streamlit | Simple Linear Regression</p>', unsafe_allow_html=True)