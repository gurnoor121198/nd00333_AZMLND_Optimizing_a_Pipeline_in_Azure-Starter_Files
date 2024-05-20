# Optimizing an ML Pipeline in Azure
# GURNOOR
## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

This dataset contains data about customers info. We seek to predict people who could be potential customers according to some parameters such as marital status, education etc.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performaing model was a automl voting Ensemble algorithm.

## Scikit-learn Pipeline

Compute target- setting up the compute cluster and selecting the best compute instances according to your requirment

Preprocessing of data- Firstly, getting the data from the link provided. Removing data if there are any NA values, then getting columns into 0,1 format using which can be used for the training purpose.

Training split- Dividing the final cleaned data into training set and testing set using the appropriate ratio. The ratio 
that we choosed here was .25. 

Early termination policy- Selecting a policy so that we early terminate a run and dont waste resources and time in running that run. This policy helps resources on exploring hyperparameter combinations that have a higher chance of yielding better results. We used bandit policy as the early termination policy in which we selected 2 of the parameters evaluation_interval=3 andslack_factor=0.

Hyperparameter Sampling:
Defining a random hyperparameter sampler for Regression, setting up the ranges for 'C' and 'max_iter'. We will tune the parameters 'C' and max_iter' using RandomParameterSampling to get the best accuracy.

HyperDrive config- setting the parameters for HyperDriveConfig() to get the best best values for hyperparameters.
HyperDriveConfig(run_config=src,
      hyperparameter_sampling=ps,
      policy=policy,
      primary_metric_name='Accuracy',
      primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
      max_total_runs=20,
      max_concurrent_runs=4)

Best run- Extracting the best run using get_best_run_by_primary_metric().


## AutoML

The best performaing model was a automl voting Ensemble algorithm , combines predictions from multiple machine learning algorithms. It is commonly used for classification tasks and uses various models to get the best results in terms of accuracy.


## Pipeline comparison
Auto ml accuracy- 91.26%
Scikit-learn accuracy- 91.23%
Automl outperformed Scikit-learn pipeline. 

## Future work
In this experiment I have used experiment_timeout_minutes in AutoMLConfig() to improve the compute and better resource utilisation.
We need more data to improve the accuracy of the model. 


## Proof of cluster clean up
![
    <img width="1700" alt="Screenshot 2024-05-20 at 5 17 23 PM" src="https://github.com/gurnoor121198/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/assets/62958740/6a89dbef-af90-47c0-8c7a-9938819971e8">

](image.png)
