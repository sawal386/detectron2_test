from models import BaseModel
from trainer import ModelFineTuner
from dataset import DatasetPreparer

import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="the path to model to be used")
    parser.add_argument("--train_data_catalog_name", help="name of the training data as "
                                                          "registered in the catalog")
    parser.add_argument("--batch_size", type=int, help="the batch size")
    parser.add_argument("--im_per_batch", type=int, help="the number of image"
                        "used in each batch while training")
    parser.add_argument("--n_class", type=int, help="number of classes in the data")
    parser.add_argument("--output_dir_path", help="path to the directory where the "
                                                  "output is saved")
    parser.add_argument("--learning_rate", type=float, help='the learning rate')
    parser.add_argument("--n_iters", type=int, help="total number of iterations")
    parser.add_argument("--test_data_name", help="name of the test data", default=None)

    parser.add_argument("--raw_data_loc", help="location of the raw data files")
    parser.add_argument("--dataset_name", help="name of the dataset")
    parser.add_argument("--annotation_info", help="json file containing the annotation"
                                                  "information")
    parser.add_argument("--dataset_type")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    model = BaseModel(args.model_path, "mps")

    preparer = DatasetPreparer(args.raw_data_loc, args.dataset_name,
                                   args.dataset_type, args.annotation_info)
    name = preparer.get_catalog_name()
    fine_tuner = ModelFineTuner(model, name, args.batch_size,
                                args.im_per_batch, args.n_class, args.output_dir_path)
    fine_tuner.run_training(args.learning_rate, args.n_iters)

