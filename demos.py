import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ------------------ MODEL COMPARISON ------------------
def model_comparison(models, X_test, y_test, le):
    st.header("Model Comparison")

    metrics = []
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0)
        })

    df = pd.DataFrame(metrics)

    numeric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(
        df.style.format({col: "{:.2f}" for col in numeric_cols})
    )

    st.subheader("Confusion Matrices")
    tabs = st.tabs(models.keys())

    for name, tab in zip(models.keys(), tabs):
        with tab:
            cm = confusion_matrix(y_test, models[name].predict(X_test))
            fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
            ax.matshow(cm, cmap="Blues", alpha=0.3)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)

            ax.set_xticklabels([""] + list(le.classes_), fontsize=8)
            ax.set_yticklabels([""] + list(le.classes_), fontsize=8)
            st.pyplot(fig)

# ------------------ TRICKY PACKETS ------------------
def tricky_packets(models, X_test, y_test):
    st.header("Tricky Packets Demo: Weak Model vs Ensembles")

    n = st.slider("Number of tricky packets", 1, 10, 5)

    idx = random.sample(range(X_test.shape[0]), n)
    X_mod = X_test[idx] + np.random.normal(0, 0.5, X_test[idx].shape)
    y_true = y_test[idx]

    rows = []
    for i in range(n):
        row = {"True": y_true[i]}
        for name, model in models.items():
            row[name] = model.predict(X_mod[i].reshape(1, -1))[0]
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- Coloring logic ----------
    def color_cells(row):
        styles = []
        true_val = row["True"]
        for col in row.index:
            if col == "True":
                styles.append("")  # no color for true label
            else:
                if row[col] == true_val:
                    styles.append("background-color: #b6fcd5")  # green
                else:
                    styles.append("background-color: #fcb6b6")  # red
        return styles

    st.write("🟩 Correct prediction &nbsp;&nbsp; 🟥 Incorrect prediction")

    st.dataframe(
        df.style.apply(color_cells, axis=1)
    )

# ------------------ NOISE STRESS TEST ------------------
def noise_stress(models, X_test, y_test):
    st.header("Noise Stress Test")

    noise_levels = [0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for name, model in models.items():
        acc = []
        for n in noise_levels:
            X_noisy = X_test + np.random.normal(0, n, X_test.shape)
            acc.append(accuracy_score(y_test, model.predict(X_noisy)))
        results[name] = acc

    st.line_chart(pd.DataFrame(results, index=noise_levels))

# ------------------ MINORITY ATTACK FOCUS ------------------
def minority_attack_focus(models, X_test, y_test, attack_class=1):
    st.header("Minority Attack Detection")

    for name, model in models.items():
        preds = model.predict(X_test)
        idx = np.where(y_test == attack_class)[0]
        rate = (preds[idx] == attack_class).mean()
        st.write(f"**{name}** attack detection rate: `{rate:.2f}`")
