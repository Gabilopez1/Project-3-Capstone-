from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset, Datastore 
from azureml.data.datapath import DataPath

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

from azureml.core.dataset import Dataset
web_path="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
ds = Dataset.Tabular.from_delimited_files(path=web_path, separator=';')


print(ds.to_pandas_dataframe())

run = Run.get_context()

def clean_data(data):
    

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    
 
    y_df = x_df.pop("quality")
    
    return x_df, y_df

x, y = clean_data(ds)

print(x.describe())

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok =True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()


