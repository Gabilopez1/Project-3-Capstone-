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
import numpy as np
import os
import joblib, pickle
from azureml.core import Model



def init():
    global model 
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best-automl-run.pkl')
    print("Found model:", os.path.isfile(model_path)) #To check whether the model is actually present on the location we are looking at
    model = joblib.load(model_path)
   
def run(data):
    try:
        #data = np.array(json.loads(data))
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
