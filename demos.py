import numpy as np
import pandas as pd
import streamlit as st
import random

def tricky_packets_demo(models, X_test, y_test):
    num_packets = st.slider("Number of tricky packets", 1, 10, 5)
    idx = random.sample(range(X_test.shape[0]), num_packets)

    X_mod = X_test[idx] + np.random.normal(0, 0.5, X_test[idx].shape)
    y_true = y_test[idx]

    results = []
    for i in range(num_packets):
        row = {"True": y_true[i]}
        for name, model in models.items():
            row[name] = model.predict(X_mod[i].reshape(1, -1))[0]
        results.append(row)

    st.dataframe(pd.DataFrame(results))
