"""Automatically crop UI sketch scans from folder and moves them to output folder."""

# import the necessary packages
import argparse
import os
from os.path import basename, splitext

import cv2
from imutils import paths


def find_contours(image):
    """Find contours from the scanned image

    Arguments:
        image {Image} -- [Scanned image read by OpenCV]

    Returns:
        List -- [List of contours]
    """
    # settings
    gaussian_kernel = (3, 3)
    gaussian_sigma_x = 0

    canny_threshold1 = 10
    canny_threshold2 = 250

    structuring_element_kernel = (3, 3)

    # grayscale the image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, gaussian_kernel, gaussian_sigma_x)

    # detect edges in the image
    edged_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)

    # construct and apply a closing kernel to 'close' gaps between 'white' pixels
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, structuring_element_kernel)
    closed = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, structuring_element)

    # find contours (i.e. the 'outlines') in the image
    (_, contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return the contours
    return contours


class Extractor:
    """
    Extract UI elements sketches from Questionnaire
    """

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # # Create necessary subfolders in output folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract(self):
        """
        Extract UI element sketches from image
        """
        # settings
        crop_offset = 200
        bounding_rectangle_width_threshold = 50
        bounding_rectangle_height_threshold = 50
        contour_outline_offset = 50
        countour_threshold = 700

        file_list = paths.list_images(self.input_folder)

        for filename in file_list:

            detected = 0

            # load the image
            loaded_image = cv2.imread(filename)

            # get image shape
            image_height, image_width, _ = loaded_image.shape

            # crop the right side where we have sketches (half width - offset)
            image = loaded_image[:image_height, ((image_width // 2) - crop_offset) : image_width]

            contours = find_contours(image)

            # if number of detected contours > threshold (happens only in consent form page)
            if len(contours) > countour_threshold:
                continue
            else:
                # loop over the contours
                for contour in contours:

                    # approximate the contour
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                    # process only the rectangles (polygon with 4 points)
                    if len(approx) == 4:

                        start, end, width, height = cv2.boundingRect(contour)

                        # if the image is too small then ignore
                        if (
                            width < bounding_rectangle_width_threshold
                            or height < bounding_rectangle_height_threshold
                        ):
                            continue

                        # offset the found contour to compensate the rectangular outline
                        [ymin, ymax] = [
                            end + contour_outline_offset,
                            end + height - contour_outline_offset,
                        ]
                        [xmin, xmax] = [
                            start + contour_outline_offset,
                            start + width - contour_outline_offset,
                        ]

                        # mark the contour as region of interest and save it in output folder
                        roi = image[ymin:ymax, xmin:xmax]

                        output_filename = f"{splitext(basename(filename))[0]}-{str(detected)}.jpg"
                        out_file = os.path.join(self.output_folder, output_filename)

                        cv2.imwrite(out_file, roi)
                        detected += 1

            print(f"Processed {filename}..")

        print("Data extraction successful...")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="""Automatically crop UI sketch scans from folder
                        and moves them to output folder."""
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

    ARGS = PARSER.parse_args()

    INPUT_FOLDER = ARGS.input_folder
    INPUT_FOLDER = INPUT_FOLDER.strip(os.sep)
    print(f"Input folder: {INPUT_FOLDER}")

    OUTPUT_FOLDER = ARGS.output_folder
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = f"{INPUT_FOLDER}_processed"
    else:
        OUTPUT_FOLDER = OUTPUT_FOLDER.strip(os.sep)

    print(f"Output folder: {OUTPUT_FOLDER}")

    EXTRACTOR = Extractor(INPUT_FOLDER, OUTPUT_FOLDER)

    EXTRACTOR.extract()
