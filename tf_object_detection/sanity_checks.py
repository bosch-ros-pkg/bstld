#!/usr/bin/env python
"""
Visual check of tfrecords. Quick way to visually rule out
some possible flaws.

python visual_inspection.py path/to/train.tfrecords
"""

import os
import sys

import cv2
import tensorflow as tf
import tqdm

from bstld.tf_object_detection.utils import ir


def _parse_record(record):
    example = tf.train.Example()
    example.ParseFromString(record)
    sample = {}
    image = example.features.feature['image/encoded'].bytes_list.value[0]
    sample['height'] = example.features.feature['image/height'].int64_list.value[0]
    sample['width'] = example.features.feature['image/width'].int64_list.value[0]
    sample['filename'] = example.features.feature['image/filename'].bytes_list.value[0]
    sample['xmin'] = example.features.feature['image/object/bbox/xmin'].float_list.value
    sample['xmax'] = example.features.feature['image/object/bbox/xmax'].float_list.value
    sample['ymin'] = example.features.feature['image/object/bbox/ymin'].float_list.value
    sample['ymax'] = example.features.feature['image/object/bbox/ymax'].float_list.value
    sample['class_ids'] = example.features.feature['image/object/class/label'].int64_list.value
    sample['class_texts'] = example.features.feature['image/object/class/text'].bytes_list.value

    sample['image'] = tf.image.decode_png(image, channels=3).eval()
    return sample


def quick_inspection(tfrecords_file, visual=True):
    """ Visual inspection if values and images look reasonable """
    if visual:
        print('Press any key toi continue and q to quit')

    record_iterator = tf.python_io.tf_record_iterator(tfrecords_file)
    with tf.Session():
        for record in tqdm.tqdm(record_iterator, desc='Checking tfrecords'):
            sample = _parse_record(record)
            image = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)

            assert sample['xmin'], 'Sample does not have bounding boxes'

            # Min values should not be larger than max values
            assert all([min_val < max_val for min_val, max_val in
                        zip(sample['xmin'], sample['xmax'])])
            assert all([min_val < max_val for min_val, max_val in
                        zip(sample['ymin'], sample['ymax'])])

            # TF object detection requires classes to start at 1
            assert all(map(lambda x: x > 0, sample['class_ids']))

            if visual:
                for box in range(len(sample['xmin'])):
                    cv2.rectangle(
                        image,
                        (ir(sample['xmin'][box] * sample['width']),
                         ir(sample['ymin'][box] * sample['height'])),
                        (ir(sample['xmax'][box] * sample['width']),
                         ir(sample['ymax'][box] * sample['height'])),
                        (255, 255, 255),
                        1)
                print(sample['filename'])
                print('classes', sample['class_ids'])  # not ordered by anything
                print('classes', sample['class_texts'])

                cv2.imshow('image', image)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print(__doc__)
        sys.exit(-1)
    quick_inspection(sys.argv[1], visual=False)
