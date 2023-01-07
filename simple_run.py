# Used for implementation

from models import Model
from visualizer import visualize_predictions

import argparse
import os
import cv2

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="the path to model to be used")
    parser.add_argument("--test_image_loc", help="path to the folder containing "
                                                 "the test images")
    parser.add_argument("--test_image_names", default=None, nargs="+",
                        help="names of test images in test_image_loc")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    model = Model(args.model_path, "cpu")
    files_list = []
    if args.test_image_names is not None:
        files_list = [args.test_image_loc + "/" + name for name in args.test_image_names]
    else:
        files_list = [args.test_image_loc + "/" + name for name in
                      os.listdir(args.test_image_names)]

    for file in files_list:
        im = cv2.imread(file)
        output = model.make_prediction(file)
        print(output)
        meta = model.get_sample_metadata()
        print(meta)
        visualize_predictions(im, meta, output, save=True)





