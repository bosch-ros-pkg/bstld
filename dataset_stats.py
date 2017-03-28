#!/usr/bin/env python
"""
Sample script to show some numbers for the dataset.

Example usage:
    python yaml_stats.py input_yaml
"""

import sys
import logging
from read_label_file import get_all_labels


def quick_stats(input_yaml):
    """
    Prints statistic data for the traffic light yaml files.

    :param input_yaml: Path to yaml file of published traffic light set
    """

    images = get_all_labels(input_yaml)

    widths = []
    heights = []
    sizes = []

    num_images = len(images)
    num_lights = 0
    appearances = {'Green': 0, 'occluded': 0}

    for image in images:
        num_lights += len(image['boxes'])
        for box in image['boxes']:
            try:
                appearances[box['label']] += 1
            except KeyError:
                appearances[box['label']] = 1

            if box['occluded']:
                appearances['occluded'] += 1

            if box['x_max'] < box['x_min']:
                box['x_max'], box['x_min'] = box['x_min'], box['x_max']
            if box['y_max'] < box['y_min']:
                box['y_max'], box['y_min'] = box['y_min'], box['y_max']

            width = box['x_max'] - box['x_min']
            height = box['y_max'] - box['y_min']
            if width < 0:
                logging.warning('Box width smaller than one at ' + image)
            widths.append(width)
            heights.append(height)
            sizes.append(width * height)

    avg_width = sum(widths) / float(len(widths))
    avg_height = sum(heights) / float(len(heights))
    avg_size = sum(sizes) / float(len(sizes))

    median_width = sorted(widths)[len(widths) // 2]
    median_height = sorted(heights)[len(heights) // 2]
    median_size = sorted(sizes)[len(sizes) // 2]

    print('Number of images:', num_images)
    print('Number of traffic lights:', num_lights, '\n')

    print('Minimum width:', min(widths))
    print('Average width:', avg_width)
    print('median width:', median_width)
    print('maximum width:', max(widths), '\n')

    print('Minimum height:', min(heights))
    print('Average height:', avg_height)
    print('median height:', median_height)
    print('maximum height:', max(heights), '\n')

    print('Minimum size:', min(sizes))
    print('Average size:', avg_size)
    print('median size:', median_size)
    print('maximum size:', max(sizes), '\n')

    print('Labels:')
    for k, l in appearances.items():
        print('\t{k}: {l}'.format(k=k, l=l))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    quick_stats(sys.argv[1])
