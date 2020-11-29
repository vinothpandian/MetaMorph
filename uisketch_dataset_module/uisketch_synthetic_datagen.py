""" Generates UI sketch dataset using labelled UI element sketches """

import glob
import logging
import math
import os
import re
from concurrent.futures import TimeoutError

import cv2
import numpy as np
from pebble import ProcessPool

from uisketch_dataset_module.annotate import XMLGenerator
from uisketch_dataset_module.rectangle import Rectangle


class UISketch_Synthetic_Datagen:  # pylint: disable=C0103
    """ Generates UI sketch dataset using labelled UI element sketches """

    def __init__(self, directory, output_folder):
        self.directory = directory
        self.image_files = glob.glob(f"{self.directory}/**/*.jpg")
        self.labels = sorted(
            [
                d
                for d in os.listdir(self.directory)
                if os.path.isdir(os.path.join(self.directory, d))
            ]
        )

        self.output_folder = output_folder

        self.image_folder = os.path.join(output_folder, "images")
        self.annotations_folder = os.path.join(output_folder, "annotations")
        self.data_folder = os.path.join(output_folder, "data")

        self.last_file_index = 0

        if os.path.exists(self.image_folder):
            existing_files = glob.glob(f"{self.image_folder}/*.jpg")

            if existing_files:
                last_file = sorted(existing_files)[-1]
                print("Found existing data.. Continuing from last generated image..")

                self.last_file_index = int(re.findall(r"(\d+)", last_file)[-1])
                print(f"Continuing from UISketch-{self.last_file_index}")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)

    def find_position(self, width, height, canvas):
        """
        Find random xmin, ymin position of rectangle within the canvas given its
        width and height

        Arguments:
            width {int} -- Width of rectangle
            height {int} -- Height of rectangle
            canvas {Rectangle} -- Rectangle class of canvas

        Returns:
            array -- xmin and ymin position of rectangle within the canvas
        """

        xmin = np.random.randint(canvas.xmin, canvas.xmax)
        ymin = np.random.randint(canvas.ymin, canvas.ymax)

        proposed = Rectangle(xmin, ymin, width, height)

        if not canvas.bounds(proposed):
            # Recurse if the rectangle doesn't bound within the canvas
            xmin, ymin = self.find_position(width, height, canvas)

        return (xmin, ymin)

    def place_images(self, rectangles, canvas):
        """
        Place given list of rectangles within the canvas without overlaps

        Arguments:
            rectangles {array} -- List of rectangles (Rectangle class)
            canvas {Rectangle} -- Rectangle class of canvas

        Returns:
            array -- List of rectangles positioned without overlaps within the
            canvas
        """
        # CHOOSE ONE
        # Pick the first rectangle
        rectangle = rectangles[0]

        # SOLVE THE BASE CASE
        if len(rectangles) == 1:
            xmin, ymin = self.find_position(rectangle.width, rectangle.height, canvas)
            rectangle.set_position(xmin, ymin)
            return rectangles

        # BACKTRACK WITHIN LOOP
        while True:
            # DELEGATE OTHERS BY RECURSION
            positioned_rectangles = self.place_images(rectangles[1:], canvas)

            # SOLVE THE PICKED ONE
            xmin, ymin = self.find_position(rectangle.width, rectangle.height, canvas)
            rectangle.set_position(xmin, ymin)

            # BREAK FROM BACKTRACK IF IT FITS
            if not rectangle.intersects_any(positioned_rectangles):
                break

        return rectangles

    def generate_thread(self, data):
        """Thread function for generating one UI sketch dataset image

        Arguments:
            data {dict} -- dictionary containing filename, elements,
            canvas_size_factor, alternate_canvas_size
        """

        filename = data["filename"]
        chosen = data["elements"]
        factor = data["canvas_size_factor"]
        alternative_canvas_size = data["alternate_canvas_size"]

        logging.debug("Starting thread %s", filename)

        # Define filepath for image and annotation file
        image_filepath = os.path.join(self.image_folder, f"{filename}.jpg")
        annotation_filepath = os.path.join(self.annotations_folder, f"{filename}.xml")

        images = []
        element_names = []
        rectangles = []
        canvas_area = 0

        # Read the image and create rectangles class for each image
        for file in chosen:
            image = cv2.imread(file)
            height, width, _ = image.shape

            # Set canvas with a factor of the area for each element
            canvas_area += width * height * factor
            images.append(image)
            element_names.append(os.path.basename(os.path.dirname(file)))
            rectangles.append(Rectangle(width=width, height=height))

        # Define the canvas area and declare it
        canvas_size = round(math.sqrt(canvas_area))

        if canvas_size > 800:
            resized_canvas_size = alternative_canvas_size
        else:
            resized_canvas_size = canvas_size

        resize_factor = canvas_size / resized_canvas_size

        canvas = Rectangle(width=canvas_size, height=canvas_size)

        # Position the rectangles
        try:
            positioned_rectangles = self.place_images(rectangles, canvas)

            # Multiplied to 255 for white background
            canvas_image = np.ones((canvas.height, canvas.width, 3), np.uint8) * 255

            # Create annotations XML generator (depth = 3)
            xml_generator = XMLGenerator(filename, resized_canvas_size, resized_canvas_size, 3)

            # Place the UI elements sketches in the canvas image
            for j, r in enumerate(positioned_rectangles):
                canvas_image[r.ymin : r.ymax, r.xmin : r.xmax] = images[j]

                resized_bndbox = (
                    int(r.xmin / resize_factor),
                    int(r.ymin / resize_factor),
                    int(r.xmax / resize_factor),
                    int(r.ymax / resize_factor),
                )
                xml_generator.add_object(element_names[j], resized_bndbox)

            resized_image = cv2.resize(
                canvas_image,
                (resized_canvas_size, resized_canvas_size),
                interpolation=cv2.INTER_CUBIC,
            )

            grayscaled_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(grayscaled_image, (3, 3), 0)

            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            final_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # Save image
            cv2.imwrite(image_filepath, final_image)

            # Save annotation
            xml_generator.save_file(annotation_filepath)

            logging.debug("Ending thread %s", filename)
        except Exception as e:
            logging.debug("Exception at %s", filename)
            logging.debug("Exception doc:  %s", e.__doc__)
            logging.debug("Exception message:  %s", str(e))

    def generate(self, limit=10, timeout=10):
        """
        Generate UI sketch dataset images in output folder until the given limit

        Keyword Arguments:
            limit {int} -- Number of dataset images to generate (default: {10})
        """

        logging.basicConfig(
            level=logging.DEBUG,
            format="(%(threadName)-9s) %(message)s",
        )

        limit = self.last_file_index + limit

        # Dynamic format spec for file naming e.g. 001
        data = []

        for i in range(self.last_file_index, limit):
            data.append(
                {
                    "filename": f"UISketch-{i:09}",
                    # Each image can have 1 to 15 elements
                    "elements": np.random.choice(
                        self.image_files, size=round(np.random.uniform(1, 15)), replace=False
                    ),
                    "canvas_size_factor": round(np.random.uniform(2, 4)),
                    "alternate_canvas_size": np.random.randint(600, 800),
                }
            )

        print(f"Using {os.cpu_count()} processes")

        with ProcessPool() as pool:
            future = pool.map(self.generate_thread, data, timeout=timeout)

            iterator = future.result()

            while True:
                try:
                    next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("Process stopped due to timeout: %d seconds" % error.args[1])

        print(f"Generated {len(os.listdir(self.image_folder))} files out of {limit}.")
