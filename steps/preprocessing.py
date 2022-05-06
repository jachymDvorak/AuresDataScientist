import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

class Dataset():

    '''
    Class to wrap the dataset and its functions
    '''

    def __init__(self, path_to_dataset, debug_mode = False):

        self.raw = pd.read_csv(path_to_dataset,
                                      sep=";")

        if debug_mode:

            self.raw = self.raw[:100]
            print(f'Debug mode, using only {len(self.raw)} rows')

        self.raw_y = self.raw['Price']
        self.raw_X = self.raw.loc[:, self.raw.columns!='Price']

        self.train_y = None
        self.test_y = None
        self.train_X = None
        self.test_x = None

    def split(self):

        print(f'Number of data points: {len(self.raw)}')

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.raw_X,
                                                                                self.raw_y,
                                                                                test_size=0.33,
                                                                                random_state=42)
        print(f'Test X length: {len(self.test_X)}')
        print(f'Test y length: {len(self.test_y)}')
        print(f'Train X length: {len(self.train_X)}')
        print(f'Train y length: {len(self.train_y)}')

    def preprocess(self):
        '''Wrapper class'''

        self.raw_X = self.drop_unnecessary_columns(dataset = self.raw_X)
        self.raw_X = self.one_hot_encode(dataset=self.raw_X)
        self.raw_X = self.replace_nans(dataset=self.raw_X)

    def drop_unnecessary_columns(self, dataset):

        cols_to_drop = ['RN', 'Advert_Title', 'Color', 'Seats', 'Doors', 'EngineSize']

        print('Unnecessary columns dropped')

        return dataset.loc[:, ~dataset.columns.isin(cols_to_drop)]

    def one_hot_encode(self, dataset):

        categorical_columns = ['Make', 'Model', 'Body', 'Fuel', 'Gearbox']

        print('One-hot encoded categorical variables')

        return pd.get_dummies(dataset, columns=categorical_columns)

    def replace_nans(self, dataset):

        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        ct = ColumnTransformer([("imputer", imputer_mean, make_column_selector(dtype_include=np.number))])

        print('Missing values for numerical columns filled with mean')

        return ct.fit_transform(dataset)

    def generate_random_subsample_for_prediction_test(self):

        np.savetxt("data/test_prediction.csv", self.test_X[:10], delimiter=";")

