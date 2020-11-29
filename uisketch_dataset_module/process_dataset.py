""" Automatically crop labelled UI sketch elements """


import argparse
import glob
import os

import cv2


def crop_image(image, bndbox):
    """Crop given image after checking whether it fits the bounding box
     and does not exceed the original image

    Arguments:
        image {numpy.ndarray} -- Image array from cv2 imread
        bndbox {tuple} -- Tuple of bounding box (xmin, ymin, xmax, ymax)

    Returns:
        numpy.ndarray -- Cropped image
    """

    xmin, ymin, xmax, ymax = bndbox

    # Check whether image fits the image shape, if not pad the image to fit the bounding box
    if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
        image, xmin, xmax, ymin, ymax = pad_image_to_fit_bndbox(image, xmin, xmax, ymin, ymax)

    return image[ymin:ymax, xmin:xmax]


def pad_image_to_fit_bndbox(image, xmin, xmax, ymin, ymax):
    """Pad image to fit the bounding box by adding border if necessary

    Arguments:
        image {numpy.ndarray} -- Image array from cv2 imread
        xmin {int} -- x min value (left top)
        ymin {int} -- y min value (left top)
        xmax {int} -- x max value (right bottom)
        ymax {int} -- y max value (right bottom)

    Returns:
        list -- image array, x min value, y min value, x max value, y max value
    """

    top = -min(0, ymin)
    bottom = max(ymax - image.shape[0], 0)
    left = -min(0, xmin)
    right = max(xmax - image.shape[1], 0)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    ymax += -min(0, ymin)
    ymin += -min(0, ymin)
    xmax += -min(0, xmin)
    xmin += -min(0, xmin)

    return image, xmin, xmax, ymin, ymax


def crop_element(image_path, output_path):
    """Crop off the whitespace in UI sketches to capture only the UI element's sketch

    Arguments:
        image_path {string} -- File path of input image file
        output_path {string} -- File path to store the cropped image
    """
    original_image = cv2.imread(image_path)

    # morph close kernel size is 10% of image width & crop offset is 1% of width
    height, width, _ = original_image.shape
    kernel_size = int(width * 0.1)
    offset = int(width * 0.01)

    # Convert original image to grayscale for further processing
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Copy the grayscale image for later reuse
    image = grayscale_image.copy()

    # Threshold to convert image black/white - remove all grays & colors
    _, thresh_binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

    # Apply gaussian blur on thresh binary image to remove noise
    denoised_image = cv2.GaussianBlur(thresh_binary_image, (7, 7), 0)

    # Find edges in the denoised image
    edged_image = cv2.Canny(denoised_image, 10, 250)

    # Close the edge detected image to form one combined element blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blob_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find all the contours
    (_, contours, _) = cv2.findContours(blob_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick only the largest contour based on area, crop it and save it in processed folder
    if contours:
        contour = max(contours, key=cv2.contourArea)
        xmin, ymin, width, height = cv2.boundingRect(contour)
        bndbox = (xmin - offset, ymin - offset, xmin + width + offset, ymin + height + offset)

        # Identify the regions of interest and save them
        roi = crop_image(image, bndbox)

        cv2.imwrite(output_path, roi)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Automatically crop labelled UI sketch elements.")

    PARSER.add_argument(
        "-i",
        "--input",
        required=True,
        dest="input_folder",
        help="Input folder containing labelled folders of UI sketches",
    )
    PARSER.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_folder",
        help="Output folder of cropped images",
    )

    ARGS = PARSER.parse_args()

    INPUT_FOLDER = ARGS.input_folder
    INPUT_FOLDER = INPUT_FOLDER.strip(os.sep)
    print(f"Input folder: {INPUT_FOLDER}")

    OUTPUT_FOLDER = ARGS.output_folder
    OUTPUT_FOLDER = OUTPUT_FOLDER.strip(os.sep)
    print(f"Output folder: {OUTPUT_FOLDER}")

    print("Creating folder structure similar to input folder in output folder.....")
    for folder in os.listdir(INPUT_FOLDER):
        if os.path.isdir(os.path.join(INPUT_FOLDER, folder)):
            os.makedirs(os.path.join(OUTPUT_FOLDER, folder), exist_ok=True)
    print("File structure cloned in output folder.")

    FILES = glob.glob(f"{INPUT_FOLDER}/**/*.jpg")

    print(f"Cropping {len(FILES)} images....")
    for image_file in FILES:
        output_file = image_file.replace(INPUT_FOLDER, OUTPUT_FOLDER)
        crop_element(image_file, output_file)

    print("All images from input folder has been processed.")
