# churn_prediction

This is a repository for predicting churn: predict the probability of a user being inactive

## How to run

* Build docker image by the provided dockerfile: Docker/Dockerfile, for example:<br>
 * cd Docker <br>
 * "docker build -t image_name:version . <br>
* Use this image to run into docker container environment

## Training

Trains an ExtraTrees classifier. Performs hyperparameter tuning and save the best model.

python ExtraTrees_training.py --data_path --model_save_path

* data_path: path of .csv file with features
* model_save_path: path where to save the best model

## Inference

Infers on the test dataset.

python ExtraTrees_Inference.py --data_path --save_preds --save_labels --model_load_path

* data_path: path of .csv file with features
* save_preds: path of .csv file to save model's probabilities on test dataset
* save_labels: path of .csv file to save labels of test dataset
* model_load_path: path where to load the model

## Metrics

Measure classification metrics by varying decision threshold from low to high values.

python metrics.py --load_preds --load_labels

* load_preds: path of .csv file where to load model's probabilities on test dataset
* load_labels: path of .csv where to load test dataset labels 

## Feature Importance

Plots the importance of trained model's features.

python feature_importance.py --model_load_path 

* model_load_path: path where to load model

