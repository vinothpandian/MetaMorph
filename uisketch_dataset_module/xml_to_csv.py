""" Converts Pascal VOC annotations to CSV """

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

import pandas as pd

from uisketch_dataset_module.split_labels import split_labels


class XML_to_CSV:
    """ Converts Pascal VOC annotations to CSV """

    def __init__(self, annotations_folder, data_folder):
        self.annotations_folder = annotations_folder
        self.data_folder = data_folder

        self.uisketch_labels_csv = os.path.join(self.data_folder, "uisketch_labels.csv")
        self.training_labels_csv = os.path.join(self.data_folder, "train_labels.csv")
        self.test_labels_csv = os.path.join(self.data_folder, "test_labels.csv")

        os.makedirs(self.data_folder, exist_ok=True)

    def xml_to_csv(self):
        """
        Converts PASCAL VOC XML annotations to CSV file

        Returns:
            DataFrame -- Pandas dataframe object of annotations
        """

        xml_list = []
        for xml_file in glob.glob(self.annotations_folder + "/*.xml"):
            try:
                tree = ET.parse(xml_file)
            except ParseError as error:
                print(f"Error occured at {xml_file}")
                print(40 * "#")
                print(error)
                print(40 * "#")
            root = tree.getroot()
            for member in root.findall("object"):
                value = (
                    root.find("filename").text,
                    int(root.find("size")[0].text),
                    int(root.find("size")[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text),
                )
                xml_list.append(value)

        column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]

        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df

    def convert(self, test_split=0.2):
        """ Converts Pascal VOC XML annotations to CSV """
        xml_df = self.xml_to_csv()

        train_df, test_df = split_labels(xml_df, test_split=test_split)

        xml_df.to_csv(self.uisketch_labels_csv, index=None)
        train_df.to_csv(self.training_labels_csv, index=None)
        test_df.to_csv(self.test_labels_csv, index=None)

        print(f"CSV files generated at {self.data_folder}")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="Generate UI sketch dataset from labelled UI element sketches."
    )

    PARSER.add_argument(
        "-ad",
        "--annotations_directory",
        required=True,
        dest="annotations_directory",
        help="Directory containing the annotations",
    )
    PARSER.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_folder",
        help="Output folder to store generate csv data",
    )

    PARSER.add_argument(
        "-t",
        "--test-split",
        default=0.2,
        type=float,
        dest="test_split",
        help="Test split percentage (default: 0.2)",
    )

    ARGS = PARSER.parse_args()

    ANNOTATIONS_DIRECTORY = ARGS.annotations_directory
    ANNOTATIONS_DIRECTORY = ANNOTATIONS_DIRECTORY.strip(os.sep)
    print(f"Directory with annotations: {ANNOTATIONS_DIRECTORY}")

    OUTPUT_FOLDER = ARGS.output_folder
    OUTPUT_FOLDER = OUTPUT_FOLDER.strip(os.sep)
    print(f"Output folder: {OUTPUT_FOLDER}")

    TEST_SPLIT = ARGS.test_split
    print(f"Test split: {TEST_SPLIT}")

    XML_TO_CSV = XML_to_CSV(ANNOTATIONS_DIRECTORY, OUTPUT_FOLDER)

    XML_TO_CSV.convert(test_split=TEST_SPLIT)
