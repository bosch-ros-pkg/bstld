#!/usr/bin/env python3
"""
Sample script to receive traffic light labels and images
of the Bosch Small Traffic Lights Dataset.

Example usage:
    python read_label_file.py input_yaml
"""

import os
import sys
import yaml

from bstld.tf_object_detection import constants


def get_all_labels(input_yaml, riib=False, clip=True):
    """ Gets all labels within label file

    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
    """
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                         images[i]['path']))

        # There is (at least) one annotation where xmin > xmax
        for j, box in enumerate(images[i]['boxes']):
            if box['x_min'] > box['x_max']:
                images[i]['boxes'][j]['x_min'], images[i]['boxes'][j]['x_max'] = (
                    images[i]['boxes'][j]['x_max'], images[i]['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                images[i]['boxes'][j]['y_min'], images[i]['boxes'][j]['y_max'] = (
                    images[i]['boxes'][j]['y_max'], images[i]['boxes'][j]['y_min'])

        # There is (at least) one annotation where xmax > 1279
        if clip:
            for j, box in enumerate(images[i]['boxes']):
                images[i]['boxes'][j]['x_min'] = max(min(box['x_min'], constants.WIDTH - 1), 0)
                images[i]['boxes'][j]['x_max'] = max(min(box['x_max'], constants.WIDTH - 1), 0)
                images[i]['boxes'][j]['y_min'] = max(min(box['y_min'], constants.HEIGHT - 1), 0)
                images[i]['boxes'][j]['y_max'] = max(min(box['y_max'], constants.HEIGHT - 1), 0)

        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted. Since they are stored in a different
        # folder, the path also needs modifications.
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    get_all_labels(sys.argv[1])
