import json
import os

import cv2
import numpy as np
import tensorflow as tf

from src.utils import reframe_box_masks_to_image_masks

# Path to object detection inference model
PATH_TO_FROZEN_GRAPH = "models/frozen_inference_graph.pb"

# Path to labels JSON.
PATH_TO_LABELS = "models/labels.json"

# Creates sketches folder to store entered sketches
os.makedirs("api/sketches", exist_ok=True)

# Load a (frozen) Tensorflow model into memory until server dies.
DETECTION_GRAPH = tf.Graph()
with DETECTION_GRAPH.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")


def create_category_index_from_labelmap(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data


# Loading label map
CATEGORY_INDEX = create_category_index_from_labelmap(PATH_TO_LABELS)


# Detection code
def run_inference_for_single_image(image, graph):
    """Run Tensorflow inference on a given image

    Arguments:
        image {ndarray} -- Image as numpy ndarray
        graph {Graph} -- Tensorflow inference graph object

    Returns:
        dict -- Output dictionary of detected element boxes, scores, and classes
    """
    with graph.as_default():
        with tf.Session() as session:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
                "detection_masks",
            ]:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if "detection_masks" in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
                # Reframe is required to translate mask from box coordinates to
                # image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1]
                )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8
                )
                # Follow the convention by adding back the batch dimension
                tensor_dict["detection_masks"] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            # Run inference
            output_dict = session.run(
                tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
            )

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]
            if "detection_masks" in output_dict:
                output_dict["detection_masks"] = output_dict["detection_masks"][0]
    return output_dict


def detect_elements(image: np.ndarray, min_prob: float):
    """Detect UI elements from the given image

    Arguments:
        image {np.ndarray} -- CV2 image object
        min_prob {float} -- Minimum probability of predictions

    Returns:
        result {dict} -- python dict of predicted result
    """

    height, width, _ = image.shape

    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscaled_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    image_np = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, DETECTION_GRAPH)

    result = []

    detected_elements = output_dict["num_detections"]

    boxes = output_dict["detection_boxes"]
    scores = output_dict["detection_scores"]
    classes = output_dict["detection_classes"]

    for i in range(detected_elements):

        if float(scores[i]) < min_prob:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        # Get real coordinates from normalized coordinates
        (left, right, top, bottom) = (
            int(xmin * width),
            int(xmax * width),
            int(ymin * height),
            int(ymax * height),
        )

        probability_score = round(scores[i] * 100, 4)
        ui_element_name = CATEGORY_INDEX[classes[i]]["name"]

        result.append(
            {
                "name": ui_element_name,
                "position": {"x": top, "y": left},
                "dimension": {"width": bottom - top, "height": right - left},
                "probability": probability_score,
            }
        )

    return result
