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
