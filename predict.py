from steps.modelling import Model
import argparse
import numpy as np

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    parser.add_argument('-p', '--path-to-dataset', help='Path to the dataset', default='./data/test_prediction.csv')
    arguments = parser.parse_args()

    # instantiate model
    model = Model()

    # load model from pkl
    model.load_model()

    # load test data TODO: change to better method
    data = np.genfromtxt(arguments.path_to_dataset, delimiter=';')

    # predict on data
    model.predict_on_new_data(data)