import os
import pickle
from model_training import train_and_save_models

MODEL_NAMES = ["weak", "bagging", "adb", "gb", "stacking", "voting"]

def load_models(X_train, y_train, model_dir="models"):
    models = {}
    missing = False

    for name in MODEL_NAMES:
        path = f"{model_dir}/{name}_model.pkl"
        if not os.path.exists(path):
            missing = True
            break

    if missing:
        return train_and_save_models(X_train, y_train, model_dir)

    for name in MODEL_NAMES:
        with open(f"{model_dir}/{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)

    return models
