from detectron2.utils.visualizer import Visualizer
import cv2
import os


def visualize_images(image_arr, metadata, image_info, scale=1.0, prediction=False,
                     save=False, full_savename="data/predicted_images/test_image"
                     , save_format="jpg"):
    """
    visualizes an image
    :param (np.ndarray) image_arr: the image that is to be visualized
    :param (Metadata) metadata: dataset metadata (e.g class names and colors)
    :param (float) scale:
    :param (dict (str) -> Instance) image_info: information about the prediction including,
                                        bounding boxes, predicted classes, masks
    :param (bool) prediction: whether or not the image contains prediction info
    :param (bool) save: whether or not to save the image
    :param (str) full_savename: the path and the name used in saving the visualizations
    :param (str) save_format: the image format used in saving the image
    """

    vis = Visualizer(image_arr[:, :, ::-1], metadata=metadata, scale=scale)
    if prediction:
        out = vis.draw_instance_predictions(image_info["instances"])
    else:
        out = vis.draw_dataset_dict(image_info)

    out_image = out.get_image()[:, :, ::-1]

    folder_loc = "/".join(full_savename.split("/")[:-1])
    os.makedirs(folder_loc, exist_ok=True)
    if save:
        cv2.imwrite("{}.{}".format(full_savename, save_format), out_image)


