# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_dir=/home/user
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import glob
import random

import dataset_util

import xml.etree.ElementTree as ET


flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('images_dir', 'images',
                    'Name of images directory.')
flags.DEFINE_string('annotations_dir', 'xml',
                    'Name of annotations directory.')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecord')
# flags.DEFINE_integer(
#     'ratio', '7', 'Ratio to split data to train set and val set. Default is train 7/ val 3')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       image_path,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='images'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      image_path: Full path to image file
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    # img_path = os.path.join(
    #     data['folder'], image_subdirectory, data['filename'])
    # full_path = os.path.join(dataset_directory, img_path)
    full_path = image_path
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    filename = full_path.split('/')[-1]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = False  # bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            if obj['name'] not in label_map_dict:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            # truncated.append(int(obj['truncated']))
            truncated.append(0)
            # poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def background_tf_example(
        image_path,
):
    """
    Args:
      image_path: Full path to image file

    Returns:
      example: The converted tf.Example.
    """

    full_path = image_path
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    filename = full_path.split('/')[-1]
    width = image.width
    height = image.height

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(images_path, output_path, images_dir_name='images', annotation_dir_name='xml'):

    # label_map_dict = {
    #     "person": 1,
    #     "face": 2
    # }
    label_map_dict = {'person': 1, 'face': 2, 'potted plant': 3, 'tvmonitor': 4, 'chair': 5, 'microwave': 6, 'refrigerator': 7, 'book': 8, 'clock': 9, 'vase': 10, 'dining table': 11, 'bear': 12, 'bed': 13, 'stop sign': 14, 'truck': 15, 'car': 16, 'teddy bear': 17, 'skis': 18, 'oven': 19, 'sports ball': 20, 'baseball glove': 21, 'tennis racket': 22, 'handbag': 23, 'backpack': 24, 'bird': 25, 'boat': 26, 'cell phone': 27, 'train': 28, 'sandwich': 29, 'bowl': 30, 'surfboard': 31, 'laptop': 32, 'mouse': 33, 'keyboard': 34, 'bus': 35, 'cat': 36, 'airplane': 37, 'zebra': 38, 'tie': 39, 'traffic light': 40, 'apple': 41, 'baseball bat': 42, 'knife': 43, 'cake': 44, 'wine glass': 45, 'cup': 46, 'spoon': 47, 'banana': 48, 'donut': 49, 'sink': 50, 'toilet': 51, 'broccoli': 52, 'skateboard': 53, 'fork': 54, 'carrot': 55, 'couch': 56, 'remote': 57, 'scissors': 58, 'bicycle': 59, 'sheep': 60, 'bench': 61, 'bottle': 62, 'orange': 63, 'elephant': 64, 'motorcycle': 65, 'horse': 66, 'hot dog': 67, 'frisbee': 68, 'umbrella': 69, 'dog': 70, 'kite': 71, 'pizza': 72, 'fire hydrant': 73, 'suitcase': 74, 'cow': 75, 'giraffe': 76, 'snowboard': 77, 'parking meter': 78, 'toothbrush': 79, 'toaster': 80, 'hair drier': 81, 'pottedplant': 82, 'sofa': 83, 'diningtable': 84, 'motorbike': 85, 'aeroplane': 86}

    logging.info('Creating {}'.format(output_path))

    writer = tf.python_io.TFRecordWriter(output_path)

    for idx in range(len(images_path)):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(images_path))
        # xml_path = xmls_path[idx]
        image_path = images_path[idx]
        xml_path = image_path.replace(
            '/{}/'.format(images_dir_name), '/{}/'.format(annotation_dir_name))
        xml_path = xml_path.replace('.jpg', '.xml')

        if os.path.exists(xml_path):            
            # print(xml_path)            
            tree = ET.parse(xml_path)
            xml = tree.getroot()
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, image_path, label_map_dict)
            writer.write(tf_example.SerializeToString())
        else:
            continue
            tf_example = background_tf_example(image_path)
            writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    data_dir = FLAGS.data_dir

    # load list image files and xml files
    images_dir = os.path.join(data_dir, FLAGS.images_dir)
    print(data_dir)
    print(images_dir)

    images_path = glob.glob(os.path.join(images_dir, '*.jpg'))
    random.seed(42)
    random.shuffle(images_path)

    # set_name = data_dir.split(os.sep)[-1]
    if str(data_dir).endswith(os.sep):
        set_name = os.path.split(data_dir)[-2]
    else:
        set_name = os.path.split(data_dir)[-1]

    print("dataset contain: {} images".format(len(images_path)))

    tfrecord_path = os.path.join(FLAGS.output_dir, '{}.record'.format(set_name))
    print('saved data at: ', tfrecord_path)

    create_tf_record(images_path, tfrecord_path, images_dir_name=FLAGS.images_dir, annotation_dir_name=FLAGS.annotations_dir)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    tf.app.run()
