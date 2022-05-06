from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

kfold = KFold(n_splits = 5, random_state = 42, shuffle = True)

xgbr = XGBRegressor(objective = 'reg:squarederror',
                         eta = 0.1,
                         max_depth = 7,
                         reg_alpha = 0.2,
                         reg_lambda = 0.2,
                         subsample = 0.7,
                         colsample_bylevel = 0.75,
                         colsample_bytree = 0.75)
bagging = BaggingRegressor(random_state = 42, n_estimators = 200)
forest = RandomForestRegressor(random_state = 42, n_estimators = 200)
gboost = GradientBoostingRegressor(random_state = 42, n_estimators = 200)

estimators = [forest, gboost, bagging, xgbr, lgbm]

models = {}

for estimator in estimators:
    estimator_name = estimator.__class__.__name__

    model = estimator.fit(X_train, y_train)

    models[estimator_name] = model

    RMSE = round(
        np.sqrt(-cross_val_score(estimator, X_train, y_train, scoring="neg_mean_squared_error", cv=kfold)).mean(), 5)
    R2 = round(cross_val_score(estimator, X_train, y_train, scoring="r2", cv=kfold).mean(), 5)

    # print(f'RMSE {estimator_name} for dataset {idx}:', RMSE)
    # print(f'R2 score {estimator_name} for dataset {idx}:', R2)

    from mlxtend.regressor import StackingCVRegressor

stack_reg = StackingCVRegressor(regressors=(forest, gboost, bagging, lgbm, xgbr), meta_regressor=lgbm,
                                use_features_in_secondary=True, cv=kfold)
stack_reg.fit(X_train, y_train);


def blender(X):
    return ((0.35 * stack_reg.predict(X)) +
            (0.25 * models['XGBRegressor'].predict(X)) +
            (0.15 * models['BaggingRegressor'].predict(X)) +
            (0.25 * models['LGBMRegressor'].predict(X)))


models['stacker'] = stack_reg

model_dict[idx] = models

pred = blender(X_val)
blended_score = round(mean_squared_error(pred, y_val, squared=False), 0)
blended_score_r2 = round(r2_score(pred, y_val), 5)

print(f'MODEL NUMBER {idx}')
print(
    f'Train set: {X_train.shape[0]}; {y_train.shape[0]}\nTest set: {X_test.shape[0]}; {y_test.shape[0]}\nValidation set: {X_val.shape[0]}; {y_val.shape[0]}')
print(f'\nRMSE of the blender is {blended_score}\nr2 is {blended_score_r2}\n\n')


class Model:

    def __init__(self, dataset, trained_model = None):

        dataset = dataset
        trained_model = trained_model

    def train(self):
        pass

    def predict(self):
        pass

    def load_from_pickle(self):
        pass

    def save_model(self):
        pass
