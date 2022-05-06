import pandas as pd
import argparse
from steps.preprocessing import Dataset
from steps.modelling import Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image processing pipeline')

    parser.add_argument('-p', '--path-to-dataset', help='Path to the dataset', default='./data/cars.csv')

    arguments = parser.parse_args()

    path_to_dataset = arguments.path_to_dataset

    dataset = Dataset(path_to_dataset = path_to_dataset)

    dataset.preprocess()

    dataset.split()

    model = Model(dataset = dataset)