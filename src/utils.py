import shutil
from typing import List
from uuid import uuid1

import cv2
import numpy as np
import tensorflow as tf
from fastapi import UploadFile

from src.models import PredictionResponse


def preprocess(image: np.ndarray):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_not(thresh_binary_image)

    desired_size = 640
    old_size = image.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    image_np = cv2.cvtColor(new_im, cv2.COLOR_GRAY2BGR)

    return image_np, top, left, ratio


def postprocess(result: List[PredictionResponse], top: int, left: int, ratio: float):

    for annotation in result:
        xmin = annotation["position"]["y"] - left
        ymin = annotation["position"]["x"] - top
        w = annotation["dimension"]["height"]
        h = annotation["dimension"]["width"]

        xmax = xmin + w
        ymax = ymin + h

        xmin = int(xmin / ratio)
        ymin = int(ymin / ratio)
        xmax = int(xmax / ratio)
        ymax = int(ymax / ratio)

        annotation["position"]["x"] = xmin
        annotation["position"]["y"] = ymin
        annotation["dimension"]["width"] = xmax - xmin
        annotation["dimension"]["height"] = ymax - ymin

    return result


def store_sketch(image: UploadFile):
    filename = uuid1()
    image_path = f"./sketches/{filename}.jpg"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return image_path


# Utility function copied from TensorFlow Object Detection API
# Original filepath: objection_detection/utils/ops
def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.

    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """

    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat([tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_ind=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            extrapolation_value=0.0,
        )

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32),
    )

    return tf.squeeze(image_masks, axis=3)
