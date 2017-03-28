#!/usr/bin/env python
"""
Quick sample script that displays the traffic light labels within
the given images.
If given an output folder, it draws them to file.

Example usage:
    python write_label_images input.yaml [output_folder]
"""
import sys
import os
import cv2
from read_label_file import get_all_labels


def ir(some_value):
    """Int-round function for short array indexing """
    return int(round(some_value))


def show_label_images(input_yaml, output_folder=None):
    """
    Shows and draws pictures with labeled traffic lights.
    Can save pictures.

    :param input_yaml: Path to yaml file
    :param output_folder: If None, do not save picture. Else enter path to folder
    """
    images = get_all_labels(input_yaml)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for i, image_dict in enumerate(images):
        image = cv2.imread(image_dict['path'])
        if image is None:
            raise IOError('Could not open image path', image_dict['path'])

        for box in image_dict['boxes']:
            cv2.rectangle(image,
                          (ir(box['x_min']), ir(box['y_min'])),
                          (ir(box['x_max']), ir(box['y_max'])),
                          (0, 255, 0))

        cv2.imshow('labeled_image', image)
        cv2.waitKey(10)
        if output_folder is not None:
            cv2.imwrite(os.path.join(output_folder, str(i).zfill(10) + '_'
                        + os.path.basename(image_dict['path'])), image)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    label_file = sys.argv[1]
    output_folder = None if len(sys.argv) < 3 else sys.argv[2]
    show_label_images(label_file, output_folder)
