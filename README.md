*Disclaimer: I only spent approx 4 hours on the task, so it definitely is not perfect and is just a demo*

## How to use

After cloning the repository, install the requirements from `requirements.txt`.

Run `train.py -d` (with optional arguments `-d` for turning on debug mode (only using 1k rows)) and `-p` as path to the dataset (default path works out of the box).

Run `predict.py` with optional argument `-p` as path to the prediction dataset (default path works out of the box, a random sample was generated).

Model is also automatically saved in the `model` folder.

## Results on whole dataset of best model (BaggingRegressor - scikit-learn):

**RMSE:** 4015

**R2:** 0.83

Conclusion: since train/val set has better scores (3829 RMSE & 0.87 R2) the model may overfit a bit. In a real setting, regularization could help.

## TODO's

Obviously this is not production ready, just a demo, therefore there is no pipeline to process data, since I use one-hot-encoding and categories change it would need to be implemented. 

Using the additional features of cars, currently it would add too many datapoints and no time to do that, but could improve the model. Would need to be de-listed.

Features could be added/filled (e.g. model of car mapping, including more features).

The hyperparameter optimalization could work per-type-of-model but only works with common hyperparameter amongst models.

Timing of model training could be implemented.

etc.


