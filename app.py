import streamlit as st
from data_utils import load_and_prepare_data
from model_loader import load_models
from demos import tricky_packets_demo

st.set_page_config(page_title="Ensemble Models", layout="wide")
st.title("Weak vs Ensemble Models")

X_train, X_test, y_train, y_test = load_and_prepare_data(
    "cybersecurity_intrusion_data.csv",
    target_column="attack_detected"
)

models = load_models(X_train, y_train)

mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Model Comparison", "Tricky Packets"]
)

if mode == "Tricky Packets":
    tricky_packets_demo(models, X_test, y_test)
