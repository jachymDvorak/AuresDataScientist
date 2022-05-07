from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
import pickle
import os
import numpy as np

class Model():
    '''
    Class to wrap the model and its functions
    '''

    def __init__(self, dataset = None, trained_model = None):

        self.dataset = dataset
        self.trained_model = trained_model

    def select_best_model(self, model_dict):

        best_estimator = min(model_dict, key=model_dict.get)

        print(f'Best estimator {best_estimator.__class__.__name__} with RMSE of {model_dict[best_estimator]}')

        return best_estimator

    def optimize_hyperparameters(self, kfold):

        model_name = self.trained_model.__class__.__name__

        print(f'Optimizing hyperparameters for model {model_name}')

        fit_gs = GridSearchCV(self.trained_model,
                             {'n_estimators': [100, 200, 400]},
                             cv=kfold).fit(self.dataset.train_X, self.dataset.train_y)

        model = fit_gs.best_estimator_

        print(f'Best hyperparameters are: {fit_gs.best_params_}')

        return model

    def train(self):

        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        xgbr = XGBRegressor(objective='reg:squarederror')
        bagging = BaggingRegressor(random_state=42, n_estimators=200, verbose=0)
        forest = RandomForestRegressor(random_state=42, n_estimators=200, verbose=0)
        gboost = GradientBoostingRegressor(random_state=42, n_estimators=200, verbose=0)

        estimators = [xgbr, forest, gboost, bagging]

        models = {}

        X_train = self.dataset.train_X
        y_train = self.dataset.train_y

        for estimator in estimators:

            estimator_name = estimator.__class__.__name__

            print(f'Training model {estimator_name}')

            model = estimator.fit(X_train, y_train)

            RMSE = round(
                np.sqrt(
                    -cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kfold)).mean(),
                5)
            R2 = round(cross_val_score(model, X_train, y_train, scoring="r2", cv=kfold).mean(), 5)

            models[model] = RMSE

            print(f'RMSE {estimator_name}:', RMSE)
            print(f'R2 score {estimator_name}:', R2)

        self.trained_model = self.select_best_model(models)

        self.trained_model = self.optimize_hyperparameters(kfold)

    def predict_on_dataset(self, data):

        estimator = self.trained_model

        return estimator.predict(data)

    def eval(self):

        pred = self.predict_on_dataset(data=self.dataset.test_X)

        test_mse = round(mean_squared_error(pred, self.dataset.test_y, squared=False), 0)
        test_r2 = round(r2_score(pred, self.dataset.test_y), 5)

        print(f'Test dataset MSE: {test_mse}')
        print(f'Test dataset R2: {test_r2}')

    def predict_on_new_data(self, data):

        estimator = self.trained_model

        for i, datapoint in enumerate(data, start = 1):
            pred = estimator.predict(datapoint.reshape(1, -1))
            print(f'Prediction for ID {i} is price {int(pred[0])} EUR.')


    def save_model(self):

        path = './model'

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, 'best_model.pickle'), 'wb') as f:
            pickle.dump(self.trained_model, f)

        print(f'Best model saved to {path}')

    def load_model(self, path = 'model/best_model.pickle'):

        with open(path, 'rb') as f:
            self.trained_model = pickle.load(f)

        print(f'Model {self.trained_model.__class__.__name__} loaded')
