"""
Create classifier dataset from close cropped labelled images
"""

import argparse
import os

import cv2
import numpy as np
from imutils import paths


def canvas_with_image_in_center(image_path, dimension):
    """
    Creates a square canvas of given dimension and places the image in the
    center

    Arguments:
        image_path {string} -- Path of the image
        dimension {int} -- Desired dimension of output image

    Returns:
        image -- Processed image
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    canvas_size = max(height, width)

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    xmin = canvas_size // 2 - (width // 2)
    ymin = canvas_size // 2 - (height // 2)
    canvas[ymin : ymin + height, xmin : xmin + width] = image

    canvas = cv2.resize(canvas, dsize=(dimension, dimension), interpolation=cv2.INTER_CUBIC)

    return canvas


def classifier_datagen(input_folder, output_folder, dimension):
    """
    Create classifier dataset from close cropped labelled images

    Arguments:
        input_folder {string} -- Path of folder with close cropped labelled
                                    images
        output_folder {string} -- Path of folder to store processed images
        dimension {int} -- Desired dimension of output image
    """
    for image_path in paths.list_images(input_folder):
        canvas = canvas_with_image_in_center(image_path, dimension)
        output_path = image_path.replace(input_folder, output_folder)
        cv2.imwrite(output_path, canvas)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="""Automatically crop UI sketch scans from folder
                        and moves them to labelled folder."""
    )

    PARSER.add_argument(
        "-i",
        "--input",
        required=True,
        dest="input_folder",
        help="Input folder containing UI sketch scans",
    )
    PARSER.add_argument(
        "-o",
        "--output",
        default=None,
        dest="output_folder",
        help="Output folder with cropped and labelled images",
    )
    PARSER.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=1000,
        dest="dimension",
        help="Desired dimension of processed images (default: 1000)",
    )

    ARGS = PARSER.parse_args()

    INPUT_FOLDER = ARGS.input_folder
    INPUT_FOLDER = INPUT_FOLDER.strip(os.sep)
    print(f"Input folder: {INPUT_FOLDER}")

    OUTPUT_FOLDER = ARGS.output_folder
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = f"{INPUT_FOLDER}_processed"
    else:
        OUTPUT_FOLDER = OUTPUT_FOLDER.strip(os.sep)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    DIMENSION = ARGS.dimension
    print(f"Output image dimensions : {DIMENSION}x{DIMENSION}")

    # Create similar directory structure as input folder in output folder
    DIRECTORIES = os.listdir(INPUT_FOLDER)
    for directory in DIRECTORIES:
        directory_path = os.path.join(INPUT_FOLDER, directory)
        if os.path.isdir(directory_path):
            os.makedirs(os.path.join(OUTPUT_FOLDER, directory), exist_ok=True)

    print("Processing images...")
    classifier_datagen(INPUT_FOLDER, OUTPUT_FOLDER, DIMENSION)
