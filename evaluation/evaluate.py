"""
Script for comparing models.

The script should be model and framework independent but does
require tensorflow and tensorflow models to be installed

class_ids:
    0: off
    1: green
    2: yellow
    3: red
"""

import json
import os

import numpy
import matplotlib.pyplot as plt

from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluation

import constants


def precision_recall_figure(precisions, recalls, json_out_path, show=True):
    """ Quick precision recall curve
    Args:
        precisions -> dict: keys are colors strings, each contains a list of precisions
                            E.g.: {'green': [1.0, .99, .72, .33, .22, .1]}
        recalls -> dict: keys are colors strings, each contains a list of recalls
        json_out_path -> str: Curves will be stored as svg instead of json
        show -> bool: If to visually show the PR curve
    """
    figure_path = os.path.splitext(json_out_path)[0] + '.svg'

    plt.figure(1)
    plt.suptitle('BSTLD precision/recall curves')
    for i, key in enumerate(precisions):
        if len(precisions) > 1:
            plt.subplot(int('22' + str(i + 1)))  # 22 grid structure, i+1 index
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()
        plt.plot(recalls[key], precisions[key])
        plt.title(key)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figure_path)
    plt.show()


def evaluate(labels, results, json_output_path, class_agnostic=False):
    """ Evaluates traffic light detections based on the BSTLD repo results
    NOTE This skips empty frames!
    Args:
        Labels and results dicts are loosely tf object_detection
        labels -> iterable of dicts:
            'ymin': list of ymin values, relative to size of image, e.g. [.2, .3, .22, .34]
            'xmin': list of xmin values, relative to size of image
            'ymax': list of ymax values, relative to size of image
            'xmax': list of xmax values, relative to size of image
            'class_labels': list of class ids, e.g. [0, 0, 3, ..]
                0: off, 1: green, 2: yellow, 3: red
        results -> iterable of dicts:
            'detection_boxes': [[xmin, ymin, xmax, ymax], ....]
            'detection_scores']: e.g. [.99, .94, .87, .70, .1, .09, ...]
            'detection_classes': list of class ids, e.g. [0, 0, 3, ..]
        json_output_path -> str: Where to output the summary
        class_agnostic: If True, all detections count
    """
    assert json_output_path.endswith('.json'), 'json_output_path not ending in json'

    evaluator = ObjectDetectionEvaluation(
        num_groundtruth_classes=len(constants.EVAL_CATEGORIES),
        matching_iou_threshold=.5, use_weighted_mean_ap=True, label_id_offset=0)
    for result_id, (label, result) in enumerate(zip(labels, results)):
        gt_boxes = numpy.array(list(zip(
            label['ymin'], label['xmin'], label['ymax'], label['xmax'])))

        if class_agnostic:
            label['class_labels'] = [0] * len(label['class_labels'])
            result['detection_classes'] = [0] * len(result['detection_classes'])

        if not label['class_labels']:
            continue

        assert all(map(lambda x: 0 <= x < 4, label['class_labels'])), 'Labels out of range [0,3]'
        assert all(map(lambda x: 0 <= x <= 1, label['xmin'])), 'Label coordinates need to be scaled'
        evaluator.add_single_ground_truth_image_info(
            image_key=result_id,
            groundtruth_boxes=numpy.array(gt_boxes),
            groundtruth_class_labels=numpy.array(label['class_labels']))

        assert all(map(lambda x: 0 <= x < 4, result['detection_classes'])),\
            'Result classes outside [0,3]'
        evaluator.add_single_detected_image_info(
            image_key=result_id,
            detected_boxes=numpy.array(result['detection_boxes']),
            detected_scores=numpy.array(result['detection_scores']),
            detected_class_labels=numpy.array(result['detection_classes']))

    eval_dict = vars(evaluator.evaluate())

    output_dict = {}
    output_dict['weighted_mean_ap'] = eval_dict['mean_ap']
    filtered_aps = list(filter(lambda x: not numpy.isnan(x),
                               eval_dict['average_precisions']))
    output_dict['mean_ap'] = sum(filtered_aps) / len(filtered_aps)
    output_dict['mean_aps'] = filtered_aps
    output_dict['num_classes'] = len(filtered_aps)

    output_dict['recalls'] = {}
    output_dict['precisions'] = {}
    for cindex in range(len(filtered_aps)):
        class_name = constants.EVAL_CATEGORIES[cindex]
        output_dict['recalls'][class_name] = list(eval_dict['recalls'][cindex])
        output_dict['precisions'][class_name] = list(eval_dict['precisions'][cindex])

    with open(json_output_path, 'w') as json_handle:
        json.dump(output_dict, json_handle)
    precision_recall_figure(output_dict['precisions'], output_dict['recalls'], json_output_path)
