#!/usr/bin/env python3
"""
Creates full-image tfrecords to use the Bosch Small Traffic Lights Dataset
with the Tensorflow Object Detection API.

The training set is split into training and validation. Tfrecords are created
for a training, validation, and test set. Labels are grouped by their respective
colors to simplify training and because the test-set does not contain any arrows.

Depending on the training method, you may want to look into creating random crops
from the images which can increase training performance due to translated inputs.
The tfrecords come without any image augmentation.

The created tfrecords will be about 18GB.

Usage:
    In the folder with the extracted traffic lights dataset, run
    python /path/to/this/file/to_tfrecords.py
    and it will create the tfrecords there.

The path of the annotation files, tfrecords, and dataset folder can be specified.
Note that this is a tutorial file. There are only few checks and no logging.
"""

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import os
from random import shuffle

import cv2
import tensorflow as tf
import tqdm

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
from object_detection.utils import dataset_util
from bstld.read_label_file import get_all_labels
from bstld.tf_object_detection import constants


def label_id(label_string):
    """ For detections without classification """
    # For object proposals only, you could return 1
    return constants.TF_ID_MAP[constants.SIMPLIFIED_CLASSES[label_string]]


def modified_label_string(label_string):
    """ To simplify the problem, training classes are grouped by color """
    return constants.SIMPLIFIED_CLASSES[label_string].encode('utf8')


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """ [{'a': 0, 'b':3}, {'a': 3, 'b':5}] --> {'a': [0, 3], 'b': [3, 5]}"""
    assert isinstance(list_of_dicts, list)
    dict_lists = defaultdict(list)
    for some_dict in list_of_dicts:
        for key, value in some_dict.items():
            dict_lists[key].append(value)
    return dict_lists


def clip(some_value):
    """ Clip values outside [0, 1]. float -> float """
    # Just in case some very eager annotators detected lights outside the image. It happens
    return max(0, min(some_value, 1))


def create_object_detection_tfrecords(labels, tfrecords_path, dataset_folder, set_name=''):
    """ Creates a tfrecord dataset specific to tensorflow/models/research/objection_detection
    params:
        labels: list of annotations as defined in annotation yamls
        tfrecords_path: output path to create tfrecords
        dataset_folder: path to bstld folder, must include rgb directory
    """

    shuffle(labels)
    writer = tf.python_io.TFRecordWriter(tfrecords_path)
    for label in tqdm.tqdm(labels, desc='Creating {}-set'.format(set_name)):
        image_path = os.path.join(dataset_folder, label['path'])
        image = cv2.imread(image_path)
        if image is None:
            print('Did you extract the training, validation, and additional images?')
            raise IOError('Missing: {}'.format(image_path))
        height, width, _ = image.shape

        boxes = list_of_dicts_to_dict_of_lists(label['boxes'])
        classes = boxes['label']
        xmin = list(map(lambda x: clip(x / float(width)), boxes['x_min']))
        ymin = list(map(lambda y: clip(y / float(height)), boxes['y_min']))
        xmax = list(map(lambda x: clip(x / float(width)), boxes['x_max']))
        ymax = list(map(lambda y: clip(y / float(height)), boxes['y_max']))

        assert len(xmin) == len(xmax) == len(ymin)
        assert len(ymax) == len(classes) == len(label['boxes'])

        if not classes:
            continue  # We don't need empty images, there are enough negatives

        _, image = cv2.imencode('.png', image)  # Assuming that works
        image = image.tostring()
        sha256 = hashlib.sha256(image).hexdigest()
        image_format = 'png'
        complete_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(sha256.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(
                list(map(modified_label_string, classes))),
            'image/object/class/label': dataset_util.int64_list_feature(
                list(map(label_id, classes))),
        }))
        writer.write(complete_example.SerializeToString())

    writer.close()


def split_train_labels(train_labels):
    # one entry for each image in a folder/video to check their sizes later
    train_videos = [train_label['path'].split('/')[-2] for train_label in train_labels]
    # NOTE Because set order is not guaranteed (and we want to support different Python versions)
    video_dict = OrderedDict().fromkeys(train_videos)
    video_lengths = [train_videos.count(video) for video in video_dict.keys()]
    # The first three videos are used for the validation set.
    # Note that this may not be a completely clean validation set as the sequences
    # were captured independently but may be on the same day and are taken within
    # the same general area. This split is for object detection demonstation
    # purposes only. For clean dataset separation, the sequences would need to be
    # recorded on separate days and preferably in different areas.
    #
    # validation samples: 933, training samples: 4160 (+215 additional)
    num_valid_samples = sum(video_lengths[:3])
    return train_labels[num_valid_samples:], train_labels[:num_valid_samples]


def create_datasets(config):
    """ Splits labels and creates datasets """
    train_labels = get_all_labels(config['train_yaml'])
    test_labels = get_all_labels(config['test_yaml'])

    if config['additional_yaml']:
        additional_labels = get_all_labels(config['additional_yaml'])

    # Split training labels into training and validation for "more correct" validation
    train_labels, valid_labels = split_train_labels(train_labels)
    train_labels.extend(additional_labels)  # add unappealing images to training set

    if not os.path.isdir(config['dataset_folder']) or\
            not os.path.isdir(os.path.join(config['dataset_folder'], 'rgb')):
        print('Dataset_folder needs to contain extracted dataset, including the rgb folder')
        print('{} does not fulfill those requirements'.format(config['dataset_folder']))

    create_object_detection_tfrecords(
        train_labels, config['train_tfrecord'], config['dataset_folder'], 'train')
    create_object_detection_tfrecords(
        valid_labels, config['valid_tfrecord'], config['dataset_folder'], 'valid')
    create_object_detection_tfrecords(
        test_labels, config['test_tfrecord'], config['dataset_folder'], 'test')

    print('Done creating tfrecords')


def parse_args():
    """ Command line args to tfrecords creation config """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_yaml', default='train.yaml',
                        help='Path to train.yaml')
    parser.add_argument('--test_yaml', default='test.yaml',
                        help='Path to test.yaml')
    parser.add_argument('--additional_yaml', default='additional_train.yaml',
                        help='Path to train_additional.yaml')
    parser.add_argument('--dataset_folder', default='.',
                        help='Path to dataset folder')
    parser.add_argument('--train_tfrecord', default='train.tfrecords',
                        help='Path to train.tfrecord')
    parser.add_argument('--valid_tfrecord', default='valid.tfrecords',
                        help='Path to valid.tfrecord')
    parser.add_argument('--test_tfrecord', default='test.tfrecords',
                        help='Path to test.tfrecord')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    config = parse_args()
    create_datasets(config)
