## Bosch Small Traffic Lights Dataset

This repository contains some scripts to get started with the Bosch Small Traffic Lights Dataset (BSTLD).
Contributions are very welcome. Simply create a pull request.

### Dataset
The dataset can be downloaded [here](https://hci.iwr.uni-heidelberg.de/node/6132). A preview of the dataset is available on YouTube by clicking on the image.

[![BSTLD Preview](https://github.com/bosch-ros-pkg/bstld/blob/master/images/dataset_sample.jpg)](https://youtu.be/P7j6XFmImAg)

Instructions on how to unzip *.zip.00X files can, for example, be found at https://hiro.bsd.uchicago.edu/node/3168
Update label files are in the label_files folder.

To convert Bosch Small Traffic Lights Dataset Annotations to Pascal VOC
Format

```
 python bosch_to_pascal.py input_yaml out_folder
```

### Sample Detections

A sample detection based on an adapted Yolo v1 model run on crops can be viewed at
[![Sample Detector View](https://github.com/bosch-ros-pkg/bstld/blob/master/images/yolo_detection_sample.jpg)](https://youtu.be/EztVEj2KnXk)]

### Results

| Method | Execution time | weighted mAP | mAP | Off | Green | Yellow | Red | External data | Link |
| ------ | -------------- | ------------ | --- | --- | ----- | ------ | --- | ------------- | ---- |
| Baseline | <100 ms | 0.36 |  |  |  |  |  | no |https://ieeexplore.ieee.org/document/7989163/|
| Hierarchical Deep Architecture | ~150 ms | 0.53 |  |  |  |  |  | no | https://arxiv.org/abs/1806.07987 |
| SSD Mobilenet V1 | 38 ms | 0.60 | 0.41 | 0.00 | 0.68 | 0.41 | 0.55 | no | https://github.com/bosch-ros-pkg/bstld/blob/master/tf_object_detection/configs/ssd_mobilenet_v1.config |
| Faster RCNN NAS-A | ~1560s | 0.65 | 0.43 | 0.00 |  0.71  | 0.33 | 0.66 | no | https://github.com/bosch-ros-pkg/bstld/blob/master/tf_object_detection/configs/faster_rcnn_nas.config  |

Values are self-reported. The evaluation is performed on the test-set without empty frames. For different goals, e.g. using minimal training data, using external data only, or others, new tables can be created. We specifically encourage non-conventional approaches.
Please make sure not to incorporate the test-set into your training, which includes multiple evaluations for different checkpoints of the same method. We understand that there can be larger variations between the different class average precisions, specifically due to the biased distribution. We will try to incorporate variations in results of the same method once reported.

### Citation
```
In case of publication based on this dataset, please cite
@inproceedings{behrendt2017deep,
  title={A deep learning approach to traffic lights: Detection, tracking, and classification},
  author={Behrendt, Karsten and Novak, Libor and Botros, Rami},
  booktitle={Robotics and Automation (ICRA), 2017 IEEE International Conference on},
  pages={1370--1377},
  year={2017},
  organization={IEEE}
}
```
