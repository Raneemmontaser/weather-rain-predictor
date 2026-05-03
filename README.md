# 🌦️ Weather Rain Predictor

A simple Streamlit web app that predicts whether it will rain based on weather inputs using a pre-trained SVM model.

---

## 📁 Files

```
archive/
├── weather_app.py               # Streamlit app
├── svm_model.pkl                # Pre-trained SVM model
├── scaler.pkl                   # Pre-trained StandardScaler
├── weather_forecast_data.csv    # Original dataset
├── weather_Prediction_notebook  # Jupyter notebook (training code)
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
cd YOUR_FOLDER_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run weather_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🎮 How to Use

1. Adjust the sliders for Temperature, Humidity, Wind Speed, Cloud Cover, and Pressure
2. Click **Predict**
3. The app will tell you if it will **Rain** or **Not Rain** with a probability score

---

## 🤖 Model Details

- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Training data:** 2500 weather records
- **Features:** Temperature, Humidity, Wind Speed, Cloud Cover, Pressure
- **Class balancing:** `class_weight='balanced'` to handle the 1:7 rain/no-rain imbalance
- **Scaler:** StandardScaler (fitted on training data, saved as `scaler.pkl`)

---

## 📦 Requirements

| Package | Version |
|---|---|
| streamlit | latest |
| scikit-learn | latest |
| numpy | 1.26.4 |

> ⚠️ numpy must be **1.26.4** — newer versions may cause import errors with matplotlib/sklearn on some setups.
