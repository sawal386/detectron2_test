# Contains classes for prepraring datasets to be used for detectron
import json
import sys
import os
import cv2
import random
import numpy as np

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from visualizer import visualize_images

from tqdm import tqdm


class DatasetPreparer:
    """
    class used to make raw dataset compatible with detectron2 model.
    made with the help of Detectro2 Tutorial
    (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF)

    Attributes:
        (str) dataset_name: the name of the dataset
        (str) dataset_dir: location of the file
        (str) attribute_info_file: full name of the file containing info about the raw
                                   annotations in json format
        (str) catalog_name: name used to register the dataset in the catalog
        (str) dataset_name: name of the dataset
    """

    def __init__(self, dataset_dir, dataset_name, dataset_type, attribute_info_file):

        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            print("{} is not a valid path. Please enter the correct path".format(
                dataset_dir))
            sys.exit()
        self.catalog_name = dataset_name + "_" + dataset_type
        self.dataset_name = dataset_name
        self.attribute_info_file = attribute_info_file
        self.reformatted_data = self.reformat_data()
        self.set_catalog()

    def reformat_data(self):
        """
        Re-annotate the data using json information

        :param (str) json_file_name: name of the json file
        :return: List[dict: (str) -> (dict)]
        """

        print("Reformatting the raw data to make it compatible with Detectron2")
        json_file_name = self.attribute_info_file
        if ".json" not in json_file_name:
            json_file_name = json_file_name + ".json"
        try:
            with open(json_file_name) as f:
                img_info = json.load(f)
        except FileNotFoundError:
            print("{} is not correct. Please enter the correct path".format(json_file_name))
            sys.exit()

        dataset_dicts = []
        for idx, v in enumerate(tqdm(img_info.values())):
            record = {}
            filename = os.path.join(self.dataset_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annotations = v["regions"]
            objs = []
            for _, anno in annotations.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {"bbox":[np.min(px), np.min(py), np.max(px), np.max(py)],
                       "bbox_mode": BoxMode.XYXY_ABS, "segmentation":[poly],
                       "category_id": 0,}
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def get_reformatted_data(self):
        """
        :return: reformatted_data
        """

        return self.reformatted_data

    def set_catalog(self):
        """
        set the data and metadata catalog

        :return: Metadata for the dataset
        """

        print("Saving data in catalog")
        DatasetCatalog.register(self.catalog_name, self.reformat_data)
        MetadataCatalog.get(self.catalog_name).set(thing_classes=[self.dataset_name])

    def get_metadata(self):
        """
        :return: (Metadata) the meta-data associated with the dataset
        """

        return MetadataCatalog.get(self.catalog_name)

    def visualize_reformatted_image(self, n_samples=1, full_savename=None):
        """
        visualize the images using the reformatted structure

        :param (str) full_savename: directory + name used in saving the images
        :param (int) n_samples: number of sample images
        """
        i = 1
        for d in random.sample(self.reformatted_data, n_samples):
            img = cv2.imread(d["file_name"])
            save_image = True if full_savename is not None else False
            cv2.imshow("image", img)
            visualize_images(img, self.get_metadata(), d, save=save_image,
                             full_savename=full_savename+"_"+ str(i))
            i += 1

    def get_catalog_name(self):
        """
        :return: (str) the catalog name
        """

        return self.catalog_name
