"""
Export graphs for object detection so it is easier to use.
This script goes through a checkpoint directory and freezes all
trained models. This can be done manually using tf od export_inference_graph
or for all checkpoints (and with less control) using this script.

Usage:
    python model_freezer.py some_checkpoint_directory
"""

import argparse
import os
import subprocess

# Should be on pythonpath after installing tf models object detection
from object_detection import export_inference_graph
import tensorflow as tf
import tqdm

from bstld.tf_object_detection import utils


def export_models_in_directory(train_directory):
    """ Exports object_detection models in a directory
    index_file: str, path to model.ckpt-123456.index
    """
    pipeline_config = os.path.join(train_directory, 'pipeline.config')
    index_files = utils.files_from_folder(train_directory, '.index')
    print('Found {} models to export'.format(len(index_files)))

    for index_file in tqdm.tqdm(index_files, desc='exporting detection graphs'):
        prefix_path = index_file.replace('.index', '')
        proto_dir = index_file.replace('.index', '_proto')
        if os.path.isdir(proto_dir):
            continue  # skip already exported graphs
        if not os.path.samefile(train_directory, os.path.dirname(index_file)):
            continue  # don't recursively build new protos from copied indices
        os.makedirs(proto_dir, exist_ok=True)
        subprocess.call([
            'python3', export_inference_graph.__file__,  # could also call exporter myself
            '--input_type image_tensor',
            '--pipeline_config_path', pipeline_config,
            '--trained_checkpoint_prefix', prefix_path,
            '--output_directory', proto_dir
        ])


def load_graph(frozen_graph_filename):
    """ Loads frozen graph from file"""
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_dir', help='Training directory with checkpoints', required=True)
    pargs = parser.parse_args()
    return vars(pargs)


if __name__ == '__main__':
    args = parse_args()
    export_models_in_directory(args['train_dir'])
