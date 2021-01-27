# Portuguese Red Wine Quality Data Set

This project is part of the Udacity Azure ML Nanodegree. In this project, two models were created: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. The main objetive is to compare compare the performance of both the models and deploy the best performing model.  

![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/starter_file/Project%20Flowchart.png)

## Project Setup and Installation
In order to run this project the following files need to be used:
- **automl.ipynb**: Jupyter Notebook to run the autoML experiment
- **hyperparameter_tuning.ipynb**: Jupyter Notebook to run the Hyperdrive experiment
- **train.py**. Script used in Hyperdrive
- **score.py**. Script used to deploy the model
- **winequality-red.csv**. The dataset used in this project
## Dataset

### Overview
The dataset that I am using is collection of variants of the Portuguese "Vinho Verde" red  wine. The dataset can be found on UC Irvine Machine Learning Repository on the following address https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv . This Dataset contain  physicochemical (inputs) and sensory (the output) variables.

### Task
The inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). The objetive is able to  detect what are the characteristics that make a good or bad wine.

### Access

For  Automated ML  I first upload the csv file with the Visual Studio with the Dataset and assigned it a key, then with the code of the Jupiter notebook (ws.datasets.keys()) I  load the dataset from the Workspace that match the key that I previous choose. 

For customized Hyperparameter with the Hyperdrive I use Dataset.Tabular.from_delimited_files() on  the python file Train.py, this last one is called on the Script Run Configuration on the Jupyter Notebook for the Hyperdrive.


## Automated ML

The following screenshot shows  the `automl` settings and configuration you used for this experiment.

![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/automl%20setting%20version1.PNG)
To run AutoML a set of parameters were chosen in an AutoML Configuration, for this case a timeout time of 20 minutes was used and  a maximun number of 5 iterations that could interact on paralell. The task was defined as classification, the primary metric is accuracy and the label column name "quality", which is the goal to predict the quality of wine. I also enable an early stopping policy to avoid overfitting aand the featurization was set automatically. 

### Results

The following screenshot were obtained with the `RunDetails` widget  for the automated ML model. For the first part the 4 Data Guardrail were performed.  Each iteration of the trained model was validated through cross-validation. Class balancing got alerted, founding that imbalanced data was detected, which can lead to a fasely perceived positive effect of a model's accuracy, because the input data has bias towards one class. This could be adressed on future work, there are many more normal wines than excellent or poor wines. One way to improve this is to obtained more data of the other classes.
There are also some missing features values on free sulfur dioxide and total sulfur dioxide, but I guess this didn't disturbed the result much.  The high cardinality feature detection obtained a no cardinality. After the Data Guardrail a summary of the pipeline being evaluated and the best observed score so far were shown.
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml1.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml2.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml3.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml4.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml5.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml6.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml7.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetautoml8.PNG)

From that list the best performing algoritm was Voting Ensemble with an accuracy of 0.66693, what it can also be see on the Visual Studio.

![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/widgetaccuracyautoml.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/azure%20learning%20studio%20best%20model.PNG)

The best model ID is show in this next screenshot:
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/modelidbest.PNG)

And its parameters of the best model are the followed:
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/fittedmodel%20automl1.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/fittedmodel%20automl2.PNG)

What I also noticed is that the accuracy is rather low, what made think that the issue with not balanced classes is the main factor to push the accuracy down.

## Hyperparameter Tuning
For the hyperameter tuning using Hyperdrive, the Dataset was treated with an assigned Scikit learn model of Logistic Regression, which is a  common supervised learning linear model for classification. The goal was to test and choose the Hyperparameter variables for later use in the Hyperdrive. The metric to seek for the best run was Accuracy.
The tuning of the parameter was performed by changing the values of C (Inverse of regularization strength. Smaller values cause stronger regularization) and max_iter(Maximum number of iteration to converge). I decided to use a Random Sampling to randomly select a value for each hyperparameter, the values here are given on a python file (train.py). I gave C and max_iter a search space for a discrete parameter using a choice from a list of explicit values, defined as a Python list. I also used bandit policy to stop a run if the target performance metric underperforms the best run so far by a specific margin, in this example the policy is applied for every iteration after the first five and abandons runs where the reported target metric is 0.1 or worse than the best performing run in an evaluation interval of 2, which is the frequency for applying the policy. The interval increases each time the training script logs the primary metric.
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/hyperdriveconfig1.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/hyperdriveconfig2.PNG)


### Results
The next screenshots shows the `RunDetails` widget with the 4 runs
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/hyperdrivecomplete.PNG)
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/bestrunwidgetaccuracy.PNG)


The accuracy of the best run was of  0.5836, with 0.8 value for Regularization Strengh and --150 for max iterations as you can see in the next Screenshot:
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/hyperbestrun.PNG)

Finally the best model was saved:
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/savemodelhyper.PNG)

From this graph, I can infer that with a higher max interations I could increase the level of accuracy better than the value for Regularization. Anyways the accuracy obtained on on the Automl was higher than the one obtained  choosing the parameters for the Hyperdrive.
![alt text](https://github.com/Gabilopez1/nd00333-capstone/blob/master/graphhyper.PNG)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
