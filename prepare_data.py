from dataset import DatasetPreparer
from pathlib import Path

import argparse
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_loc", help="location of the raw data files")
    parser.add_argument("--save_name", help="location where the reformatted data"
                                             "is saved", default=None)
    parser.add_argument("--save_loc", help="location where the output is saved")
    parser.add_argument("--dataset_name", help="name of the dataset")
    parser.add_argument("--annotation_info", help="json file containing the annotation"
                                                  "information")
    parser.add_argument("--dataset_type")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    savename = args.save_name
    if savename is None:
        savename = args.dataset_name + "_" + args.dataset_type

    preparer = DatasetPreparer(args.raw_data_loc, args.dataset_name,
                               args.dataset_type, args.annotation_info)
    data = preparer.get_reformatted_data()
    save_loc = Path(args.save_loc)
    name = "{}_{}".format(args.dataset_name, args.dataset_type)
    with open(save_loc / "{}.pkl".format(name), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Exported the data as: {}".format(save_loc / "{}.pkl".format(name)))

    preparer.visualize_reformatted_image(full_savename=args.save_loc+"/{}/{}".format("sample_images",
                                                                                     "image"))
    print("Prepared dataset")






