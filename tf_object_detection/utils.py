"""
List of quick helper functions
"""

import os

LOWER_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.ppm', '.gif']
IMAGE_EXTENSIONS = LOWER_IMAGE_EXTENSIONS + list(map(lambda x: x.upper(), LOWER_IMAGE_EXTENSIONS))


def ir(some_value):
    """ Because opencv wants integer pixel values"""
    return int(round(some_value))


def tir(some_value):
    """ Specific to opencv's need for explicit integer tuples
    Mostly used for (x, y) in cv2.line, circle, rectangle..."""
    return tuple(map(ir, some_value))


def _keep_extensions(files, extension):
    """ Filters by file extension, this can be more than the extension!
    E.g. .png is the extension, gray.png is a possible extension"""
    if isinstance(extension, str):
        extension = [extension]

    def one_equal_extension(some_string, extension_list):
        return any([some_string.endswith(one_extension) for one_extension in extension_list])

    return list(filter(lambda x: one_equal_extension(x, extension), files))


def files_from_folder(path, extension):
    """ Returns files within folder with given extension
    Args:
      path->str: path to input directory
      extension->str: file type to find
    Returns: List of paths
    """
    files = []
    for root, subfolders, some_files in os.walk(path):
        for some_file in some_files:
            files.append(os.path.join(root, some_file))
    print('Getting image files at {}'.format(path))
    files = _keep_extensions(files, IMAGE_EXTENSIONS)
    files = list(map(os.path.abspath, files))
    files = sorted(files)  # warning: '/asd/11.png' > '/asd/5.png' and several cases

    return files


def images_from_folder(path):
    """ Returns list of images within folder """
    return files_from_folder(path, IMAGE_EXTENSIONS)
