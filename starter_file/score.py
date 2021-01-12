# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.6 - AzureML
#     language: python
#     name: python3-azureml
# ---

# +
import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model


def init():
    global daone
    model_path = Model.get_model_path('best-automl.pkl')
    daone = joblib.load(model_path)

def run(data):
    try:
        trynn = json.loads(data)
        data = pd.DataFrame(trynn['data'])
        result = daone.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
# -


