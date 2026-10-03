import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Predicción de Depósito", layout="centered")
st.title("Predicción de Depósito Bancario")

@st.cache_resource
def load_pack():
    return load("modelo_final.joblib")

try:
    pack = load_pack()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

pipeline = pack["pipeline"]
feature_metadata = pack["feature_metadata"]
classes_ = pack.get("classes_", [])

with st.form("prediction_form"):
    inputs = {}
    
    st.subheader("Variables Numéricas")
    num_cols = st.columns(2)
    num_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "numerical":
            with num_cols[num_idx % 2]:
                med = float(meta.get("median", 0.0))
                inputs[feat] = st.number_input(
                    label=feat,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    value=med,
                    step=1.0 if med.is_integer() else 0.1
                )
            num_idx += 1
            
    st.subheader("Variables Categóricas")
    cat_cols = st.columns(2)
    cat_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "categorical":
            with cat_cols[cat_idx % 2]:
                opts = meta.get("options", [])
                inputs[feat] = st.selectbox(
                    label=feat,
                    options=opts,
                    index=0 if opts else None
                )
            cat_idx += 1
            
    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    X_new = pd.DataFrame([inputs])
    
    try:
        y_pred = pipeline.predict(X_new)[0]
        
        if y_pred == "yes":
            st.success(f"Predicción: Sí")
        else:
            st.error(f"Predicción: No")
            
    except Exception as e:
        st.error(f"Error: {e}")