## Tensorflow object detection example

### Prerequisites:
Tensorflow and models/research/object_detection need to be installed
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

### Usage:
Run in this order:
  1) to_tfrecords.py
  2) models/research/object_detection/train.py
          modify ssd_mobilenet_v1.config to use paths on your system
          Needs BSTLD downloaded and extracted first
  3) model_freezer.py to adapt models for easier loading
  4) inference.py to use trained and frozen models

Each file should come with some instructions

