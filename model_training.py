import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import streamlit as st
import os


@st.cache_resource
def train_and_save_models(X_train, y_train, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    models = {}

    models["weak"] = DecisionTreeClassifier(max_depth=1, random_state=42)
    models["weak"].fit(X_train, y_train)

    models["bagging"] = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    ).fit(X_train, y_train)

    models["adb"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    ).fit(X_train, y_train)

    models["gb"] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=1,
        random_state=42
    ).fit(X_train, y_train)

    models["stacking"] = StackingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=1)),
            ("svc", SVC(probability=True))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    ).fit(X_train, y_train)

    models["voting"] = VotingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=1)),
            ("bag", BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)),
            ("svc", SVC(probability=True))
        ],
        voting="soft"
    ).fit(X_train, y_train)

    for name, model in models.items():
        with open(f"{model_dir}/{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    return models
