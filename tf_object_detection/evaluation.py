"""
Starting short and simple to only get a few numbers
"""

import argparse

import tensorflow as tf
import tqdm

from bstld.read_label_file import get_all_labels
from bstld.evaluation.evaluate import evaluate as bstld_evaluate
from bstld.tf_object_detection import inference
from bstld.tf_object_detection import constants


def evaluate_tfrecords(frozen_graph_path, tfrecords_path, json_output_path, class_agnostic):
    """ Evaluates a training/validation/test set using AP, mAP and weighted mAP
    Args:
        frozen_graph_path->str: path to frozen trained tf_object_detection model
        yaml_path->str: path to BSTLD label file with rgb folder in same directory
        json_output_path->str: Where to store the results
        class_agnostic: Whether or not to distinguish between the different light colors
    """

    # NOTE Should use use non-python_io reading and move reading to utils
    labels = []
    with tf.Session():
        for record in tqdm.tqdm(tf.python_io.tf_record_iterator(tfrecords_path),
                                desc='Preloading annotations'):

            example = tf.train.Example()
            example.ParseFromString(record)
            label = {
                'xmin': example.features.feature['image/object/bbox/xmin'].float_list.value,
                'xmax': example.features.feature['image/object/bbox/xmax'].float_list.value,
                'ymin': example.features.feature['image/object/bbox/ymin'].float_list.value,
                'ymax': example.features.feature['image/object/bbox/ymax'].float_list.value,
                'class_labels': [
                    class_id - 1 for class_id in
                    example.features.feature['image/object/class/label'].int64_list.value],
            }
            labels.append(label)

    results = inference.inference_tfrecords(
        frozen_graph_path, tfrecords_path, json_output_path, show=False)

    results = list(map(_update_result_format, results))

    bstld_evaluate(labels, results, json_output_path, class_agnostic)


def evaluate_yaml_file(frozen_graph_path, yaml_path, json_output_path, class_agnostic):
    """ Evaluates over complete label file for correctness using AP, mAP, weightedAP
    There may be simplications, skipped files, removed biases, augmentation, ...
    in training sets.
    Args:
        frozen_graph_path->str: path to frozen trained tf_object_detection model
        yaml_path->str: path to BSTLD label file with rgb folder in same directory
        json_output_path->str: Where to store the results
        class_agnostic: Whether or not to distinguish between the different light colors
    """
    raw_labels = get_all_labels(yaml_path)
    labels = []
    for raw_label in raw_labels:
        label = {}
        label['xmin'] = [box['x_min'] / constants.WIDTH for box in raw_label['boxes']]
        label['xmax'] = [box['x_max'] / constants.WIDTH for box in raw_label['boxes']]
        label['ymin'] = [box['y_min'] / constants.HEIGHT for box in raw_label['boxes']]
        label['ymax'] = [box['y_max'] / constants.HEIGHT for box in raw_label['boxes']]
        label['class_labels'] = [box['label'] for box in raw_label['boxes']]
        label['class_labels'] = list(map(
            lambda x: constants.EVAL_ID_MAP[constants.SIMPLIFIED_CLASSES[x]],
            label['class_labels']))
        labels.append(label)

    results = inference.label_file_inference(frozen_graph_path, yaml_path,
                                             json_output_path, show=False)
    results = list(map(_update_result_format, results))

    bstld_evaluate(labels, results, json_output_path, class_agnostic)


# Remove outer array, use class ids from 0 to n
def _update_result_format(result):
    result = {key: value[0] for key, value in result.items()}
    result['detection_classes'] = list(map(lambda x: x - 1, result['detection_classes']))
    return result


def evaluate(frozen_graph_path, data_input_path, json_out_path, class_agnostic):

    if args['data_input'].endswith('.tfrecords'):
        evaluate_tfrecords(frozen_graph_path, data_input_path, json_out_path, class_agnostic)
    elif args['data_input'].endswith('.yaml'):
        evaluate_yaml_file(frozen_graph_path, data_input_path, json_out_path, class_agnostic)
    else:
        raise ValueError('Unknown data input: {}'.format(data_input_path))


def parse_args(desc=__doc__):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--frozen_graph', required=True,
                        help='Path to frozen graph')
    parser.add_argument('--data_input', required=True,
                        help='Yaml file or tfrecords')
    parser.add_argument('--json_out', default=None,
                        help='Where, if json file should be stored')
    parser.add_argument('--no_classes', action='store_true',
                        help='Class agnostic evaluation')
    pargs = parser.parse_args()
    return vars(pargs)


if __name__ == '__main__':
    args = parse_args(__doc__)
    assert args['json_out'] is not None, '--json_out needs to specify output path'
    evaluate(args['frozen_graph'], args['data_input'], args['json_out'],
             class_agnostic=args['no_classes'])
