# This program configures the detectron2 model to be used in training
import sys
import os

# import some common detectron2 utilities
from detectron2 import model_zoo #  collection of baseline models
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2

class BaseModel:
    """
    object that configures the model
    Attributes:
        (CfgNone) cfg: the configuration object
        (DefaultPredictor) predictor: object used to make predictors based on cfg
        (str) config_file_path = config file name relative to detectron2' configs
                                       directory
    """

    def __init__(self, config_file_path, device, roi_thresh=0.5, **kwargs):
        """
        :param (str) model_type: the detection architecture to be used
        """

        self.cfg = get_cfg()
        self.config_file_path = config_file_path
        self.cfg.MODEL.DEVICE = device
        self.model_name = self.__find_name()
        try:
            self.cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_path)

        except RuntimeError:
            print("{} is not a valid path. Recheck the path name".format(config_file_path))
            sys.exit()
        self.predictor = DefaultPredictor(self.cfg)

    def __find_name(self):
        """
        :return: (str) the ML model name
        """

        model_name = " ".join(self.config_file_path.split("/")[-1].split("_")[0:2])
        return model_name

    def make_prediction(self, image_path):
        """
        make predictions on an image
        :param (str) image_path: the path to the image on which predictions are made

        :return: (dict)
        """

        image = cv2.imread(image_path)

        return self.predictor(image)

    def get_sample_metadata(self):
        """

        :return: sample metadata catalog
        """
        return MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def get_model_cfg(self):
        """
        :return: the cfg file
        """

        return self.cfg

    def setup_training(self, train_data, images_per_batch, batch_size_per_image,
                       num_class, output_dir, **other_params):
        """
        configure the training parameters

        :param (int) train_data: name of the data; needs to be catalogued
        :param (int) images_per_batch: number of images used in a single batch
        :param (int) batch_size_per_image: how many ROI's are generated from each images
                                            in a batch
        :param (int) num_class: number of foreground classes
        :param (str) output_dir: location where the outputs are saved
        """

        self.cfg.DATASETS.TRAIN = (train_data, )
        self.cfg.SOLVER.IMS_PER_BATCH = images_per_batch
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)


    def assign_hyperparameters(self, lr, total_iters, **other_hyperparameters):
        """
        assigns model hyperparameters

        :param (float) lr: learning rate
        :param (int) total_iters: total number of iterations
        :param other_hyperparameters:other model hyperparameters
        """

        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.MAX_ITER = total_iters
        if "decay_steps" in other_hyperparameters:
            self.cfg.SOLVER.STEPS = other_hyperparameters["decay_steps"]



