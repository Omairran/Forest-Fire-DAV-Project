# model_comparison_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forest Fire Model Comparator", layout="centered")
st.title("🔥 Forest Fire Risk Prediction Model Comparison")

# === Upload CSV ===
uploaded_file = st.file_uploader("Upload forestfires.csv file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # === Binary Target Variable ===
    df['fire_occurred'] = (df['area'] > 0).astype(int)

    # === Features and Target ===
    features = df[['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI']]
    target = df['fire_occurred']

    # === Train/Test Split and Scale ===
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Define Models ===
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # === Train & Evaluate Models ===
    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    # === Convert to DataFrame ===
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    # === Sidebar: Model Selector ===
    selected_model = st.selectbox("Select a model to view results:", results_df["Model"])

    # === Show Selected Model's Performance ===
    st.subheader(f"📊 Performance: {selected_model}")
    st.dataframe(results_df[results_df["Model"] == selected_model].set_index("Model").style.format("{:.2f}"))

    st.subheader("📉 Confusion Matrix")
    y_pred_selected = predictions[selected_model]
    cm = confusion_matrix(y_test, y_pred_selected)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {selected_model}")
    st.pyplot(fig)

    # === Show Comparison Table ===
    st.subheader("📋 All Models Comparison")
    st.dataframe(results_df.set_index("Model").style.format("{:.2f}"))
else:
    st.info("👈 Upload your `forestfires.csv` file to get started.")
