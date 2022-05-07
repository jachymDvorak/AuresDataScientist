import argparse
from steps.preprocessing import Dataset
from steps.modelling import Model

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    parser.add_argument('-p', '--path-to-dataset', help='Path to the dataset', default='./data/cars.csv')
    parser.add_argument('-d', '--debug-mode', help='Turn on/off debug mode', action='store_true')
    arguments = parser.parse_args()

    # instantiate dataset with path to dataset
    path_to_dataset = arguments.path_to_dataset
    dataset = Dataset(path_to_dataset = path_to_dataset,
                      debug_mode = arguments.debug_mode)

    # preprocess data
    dataset.preprocess()

    # split into train and test data
    dataset.split()

    # generate random test dataset from train sample TODO: Better method
    dataset.generate_random_subsample_for_prediction_test()


    # instantiate model with dataset
    model = Model(dataset = dataset)

    # train model
    model.train()

    # evaluate model on test set
    model.eval()

    # save model as pkl
    model.save_model()