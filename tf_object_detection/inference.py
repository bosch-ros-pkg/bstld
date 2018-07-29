"""
"""

import argparse
import os
import timeit
import types

import cv2
import numpy
import tensorflow as tf
import tqdm

from bstld.read_label_file import get_all_labels
from bstld.tf_object_detection import model_freezer
from bstld.tf_object_detection import constants
from bstld.tf_object_detection import utils

# TODO move session creation out of inference for parsing single samples
#      OR replace with generators. That sounds feasible
# TODO Add some dimension asserts
# TODO json output to be implemented
# TODO docstrings
# TODO make pretty


def jsonify(some_dict):
    assert all(isinstance(some_value, numpy.ndarray) for some_value in some_dict.values())
    return {key: value.tolist() for key, value in some_dict.items()}


def suboptimal_inference_timing(frozen_graph_path, num_samples=100):
    # NOTE Stores all random samples in memory to not have an effect on inference time
    graph = model_freezer.load_graph(frozen_graph_path)
    sample_images = numpy.random.randint(
        0, 255, (num_samples, constants.HEIGHT, constants.WIDTH, 3))
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            start_time = timeit.default_timer()
            for i in range(num_samples):
                sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: sample_images[i:i + 1]})
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            print('Overall {} seconds'.format(elapsed))
            print('Per sample: {}'.format(elapsed / num_samples))


def inference_tfrecords(graph, tfrecords_path, json_out_path, show, out_folder=None, threshold=.5):
    # NOTE It would be hard to get this more inefficient
    #      The correct way is to move the visualization out of the data inference
    #      and call it for the results, one image at a time
    # NOTE The session would need to be started before the data loading so it can be active
    #      for all images and not be created for every single one (30 minute fix)

    # TODO check if this works and remove TODO
    # NOTE This has thrown a nested Session Error before which makes sense
    def images():
        with tf.Session():
            for record in tf.python_io.tf_record_iterator(tfrecords_path):

                example = tf.train.Example()
                example.ParseFromString(record)
                image = example.features.feature['image/encoded'].bytes_list.value[0]
                image = tf.image.decode_png(image, channels=3).eval()
                yield numpy.expand_dims(image, 0)

    results = data_inference(images(), graph, show=show, out_folder=out_folder, threshold=threshold)
    return results


def inference_folder(graph, folder_path, json_out_path, show, out_folder=None, threshold=.5):

    image_paths = utils.images_from_folder(folder_path)
    # NOTE This will need to be split up for large folders
    #      Move session creation to earlier point, call individually, see inference_tfrecords

    def images():
        for image_path in image_paths:
            yield numpy.expand_dims(cv2.cvtColor(cv2.imread(image_path),
                                    cv2.COLOR_BGR2RGB), 0)

    results = data_inference(images(), graph, show, out_folder=out_folder, threshold=threshold)
    return results


def image_inference(graph, image_path, json_out_path, show, out_folder=None, threshold=.5):
    if out_folder is not None:
        raise NotImplementedError('image out folder not implemented')

    image = cv2.imread(image_path)
    image = cv2.resize(image, (constants.WIDTH, constants.HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = numpy.expand_dims(image, 0)
    results = data_inference(image, graph, show, out_folder=out_folder, threshold=threshold)

    return results


def label_file_inference(graph, label_path, json_out_path, show, out_folder=None, threshold=.5):
    assert os.path.exists(label_path), 'Label path not found: {}'.format(label_path)
    labels = get_all_labels(label_path)   # Assuming 8 bit images

    def images():
        for label in labels:
            image = cv2.imread(label['path'])   # Remove copied code from image_inference
            image = cv2.resize(image, (constants.WIDTH, constants.HEIGHT))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = numpy.expand_dims(image, 0)
            yield image

    results = data_inference(images(), graph, show, out_folder=out_folder, threshold=threshold)

    return results


def draw_detected_lights(input_tensor, results_dict, min_confidence=.5):
    # TODO Currently called without min_confidence. Make command line argument
    if len(input_tensor.shape) == 4:
        if input_tensor.shape[0] != 1:
            raise ValueError('Only supposed to get one image tensor, '
                             'shape is {}'.format(input_tensor.shape))
        else:
            input_tensor = input_tensor[0]
    assert input_tensor.shape[2] == 3, 'Expecting hwc image, BGR'

    image = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    for box_number, detection in enumerate(results_dict['detection_boxes'][0]):
        if results_dict['detection_scores'][0][box_number] > min_confidence:
            cv2.rectangle(
                image,
                utils.tir((detection[1] * constants.WIDTH,
                           detection[0] * constants.HEIGHT)),
                utils.tir((detection[3] * constants.WIDTH,
                           detection[2] * constants.HEIGHT)),
                constants.CLASS_COLORS[
                    results_dict['detection_classes'][0][box_number]],
                3)
    return image


def data_inference(tensor_data, frozen_graph_path, show=False, out_folder=None, threshold=.5):
    """ Returns results """
    # NOTE Single datapoint inference at a time. Really not optimized. At all.
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
    outputs = []
    graph = model_freezer.load_graph(frozen_graph_path)
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes',
                        'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # Making single images iterable
            if not isinstance(tensor_data, (list, types.GeneratorType)):
                tensor_data = [tensor_data]

            for counter, tensor_image in tqdm.tqdm(enumerate(tensor_data), desc='Inference'):
                assert len(tensor_image.shape) == 4, 'Tensor input needs to be 4D'
                output_dict = sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: tensor_image})
                outputs.append(output_dict)
                output_dict = jsonify(output_dict)  # Everything to lists

                if show or out_folder:
                    image = draw_detected_lights(tensor_image, output_dict, threshold)
                if show:
                    cv2.imshow('image', image)
                    cv2.waitKey(show)

                if out_folder:
                    image_output_path = os.path.join(out_folder, str(counter).zfill(6) + '.png')
                    cv2.imwrite(image_output_path, image)
    return outputs


def inference(frozen_graph_path, data_input_path, json_out_path,
              show=False, out_folder=None, threshold=.5):

    if args['data_input'].endswith('.tfrecords'):
        inference_tfrecords(frozen_graph_path, data_input_path,
                            json_out_path, show, out_folder, threshold)
    elif os.path.isdir(args['data_input']):
        inference_folder(frozen_graph_path, data_input_path,
                         json_out_path, show, out_folder, threshold)
    elif args['data_input'].endswith('.png'):
        image_inference(frozen_graph_path, data_input_path,
                        json_out_path, show, out_folder, threshold)
    elif args['data_input'].endswith('.yaml'):
        label_file_inference(frozen_graph_path, data_input_path,
                             json_out_path, show, out_folder, threshold)
    else:
        raise ValueError('Unknown data input: {}'.format(data_input_path))


def parse_args(desc=__doc__):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--frozen_graph', required=True,
                        help='Path to frozen graph')
    parser.add_argument('--data_input', required=True,
                        help='Image path, folder path, or tfrecords')
    parser.add_argument('--json_out', default=None,
                        help='Where, if json file should be stored')
    parser.add_argument('--show', default=0, type=int,
                        help='Duration to show images in ms')
    parser.add_argument('--out_folder', default=None, type=str,
                        help='Folder to write images with detections')
    parser.add_argument('--threshold', default=50, type=int,
                        help='Output threshold in percent for drawn detections')
    pargs = parser.parse_args()
    return vars(pargs)


if __name__ == '__main__':
    args = parse_args()
    inference(
        frozen_graph_path=args['frozen_graph'],
        data_input_path=args['data_input'],
        json_out_path=args['json_out'],
        show=args['show'],
        out_folder=args['out_folder'],
        threshold=args['threshold'] / 100.0
    )
