# Portuguese Red Wine Quality Data Set

This project is part of the Udacity Azure ML Nanodegree. In this project, two models were created: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. The main objetive is to compare compare the performance of both the models and deploy the best performing model.  

![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/starter_file/Project%20Flowchart.png))

## Project Setup and Installation
In order to run this project the following files need to be used:
- **automl.ipynb**: Jupyter Notebook to run the autoML experiment
- **hyperparameter_tuning.ipynb**: Jupyter Notebook to run the Hyperdrive experiment
- **train.py**. Script used in Hyperdrive
- **score.py**. Script used to deploy the model
- **winequality-red.csv**. The dataset used in this project
## Dataset

### Overview
The dataset that I am using is collection variants of the Portuguese "Vinho Verde" red  wine. The dataset can be found on UC Irvine Machine Learning Repository on the following address https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv . This Dataset contain  physicochemical (inputs) and sensory (the output) variables.

### Task
The inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). The objetive is able to  detect what are the characteristics that make a good or bad wine.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
