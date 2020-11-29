"""
Generates TensorFlow records from csv annotations
"""

import io
import os
from collections import namedtuple

import contextlib2
import pandas as pd
import tensorflow as tf
from PIL import Image


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards

    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        "{}-{:05d}-of-{:05d}".format(base_path, idx, num_shards) for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


class Generate_TF_Records:
    """
    Generates TensorFlow records from csv annotations
    """

    def __init__(self, image_folder, data_folder, labels):
        self.image_folder = image_folder
        self.data_folder = data_folder
        self.labels = labels

        self.labels_file = os.path.join(self.data_folder, "labels.pbtxt")

    def label_to_int(self, row_label):
        """Converts label to int

        Arguments:
            row_label {string} -- Label name

        Returns:
            [type] -- [description]
        """
        return self.labels.index(row_label) + 1

    def split(self, df, group):
        """Split Pandas DataFrame by filename group

        Arguments:
            df {DataFrame} -- Pandas DataFrame object
            group {string} -- Group by name

        Returns:
            list -- list of data tuples with filename and object
        """
        data = namedtuple("data", ["filename", "object"])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group):
        """Create TensorFlow example file

        Arguments:
            group {string} -- group by name

        Returns:
            Example -- Tensorflow example object tf.train.Example
        """
        with tf.gfile.GFile(
            os.path.join(self.image_folder, "{}".format(group.filename)), "rb"
        ) as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode("utf8")
        image_format = b"jpg"
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        classes_text = []
        classes = []

        for _, row in group.object.iterrows():
            xmin.append(row["xmin"] / width)
            xmax.append(row["xmax"] / width)
            ymin.append(row["ymin"] / height)
            ymax.append(row["ymax"] / height)
            classes_text.append(row["class"].encode("utf8"))
            classes.append(self.label_to_int(row["class"]))

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": int64_feature(height),
                    "image/width": int64_feature(width),
                    "image/filename": bytes_feature(filename),
                    "image/source_id": bytes_feature(filename),
                    "image/encoded": bytes_feature(encoded_jpg),
                    "image/format": bytes_feature(image_format),
                    "image/object/bbox/xmin": float_list_feature(xmin),
                    "image/object/bbox/xmax": float_list_feature(xmax),
                    "image/object/bbox/ymin": float_list_feature(ymin),
                    "image/object/bbox/ymax": float_list_feature(ymax),
                    "image/object/class/text": bytes_list_feature(classes_text),
                    "image/object/class/label": int64_list_feature(classes),
                }
            )
        )
        return tf_example

    def generate_records(self, csv_path, record_name, num_shards=10):
        """Generate TensorFlow records file

        Arguments:
            csv_path {string} -- CSV file path
            record_name {string} -- TensorFlow record name

        Keyword Arguments:
            num_shards {int} -- count of shards to split Tensorflow Record
            (default: {10})
        """
        examples = pd.read_csv(csv_path)
        grouped = self.split(examples, "filename")
        output_path = os.path.join(self.data_folder, record_name)

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = open_sharded_output_tfrecords(
                tf_record_close_stack, output_path, num_shards
            )
            for index, group in enumerate(grouped):
                tf_example = self.create_tf_example(group)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

        print(f"Successfully created the TFRecords: {output_path}")

        with open(self.labels_file, "w") as file:
            for i, label in enumerate(self.labels):
                file.write(f"item {{\n  id: {i+1}\n  name: '{label}'\n}}\n\n")

        print(f"Successfully created the labels file: {self.labels_file}")
