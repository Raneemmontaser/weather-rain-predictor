import streamlit as st
import pickle

st.set_page_config(page_title="🌦️ Rain Predictor", layout="centered")
st.title("🌦️ Weather Rain Predictor")
st.markdown("Adjust the sliders and click **Predict** to see if it will rain.")

@st.cache_resource
def load_models():
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return svm, scaler

svm, scaler = load_models()

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    temp     = st.slider("🌡️ Temperature (°C)",  10.0, 35.0, 22.0, 0.1)
    humidity = st.slider("💧 Humidity (%)",       30.0, 100.0, 65.0, 0.5)
    wind     = st.slider("💨 Wind Speed (km/h)",   0.0,  20.0, 10.0, 0.1)
with col2:
    cloud    = st.slider("☁️ Cloud Cover (%)",     0.0, 100.0, 50.0, 0.5)
    pressure = st.slider("📊 Pressure (hPa)",    980.0, 1050.0, 1013.0, 0.5)

st.markdown("---")

if st.button("🔮 Predict", use_container_width=True):
    input_scaled = scaler.transform([[temp, humidity, wind, cloud, pressure]])
    prob  = svm.predict_proba(input_scaled)[0][1]
    label = "🌧️ RAIN" if prob >= 0.5 else "☀️ NO RAIN"
    color = "🔴" if prob >= 0.5 else "🟢"

    st.markdown(f"### {color} Prediction: **{label}**")
    st.progress(float(prob))
    st.caption(f"Rain probability: {prob * 100:.1f}%")
