""" Generate UISketch synthetic dataset from labelled UI element sketches. """

import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


import argparse
import os

from uisketch_dataset_module.generate_tf_record import Generate_TF_Records
from uisketch_dataset_module.uisketch_synthetic_datagen import UISketch_Synthetic_Datagen
from uisketch_dataset_module.xml_to_csv import XML_to_CSV

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate UISketch synthetic dataset from labelled UI element sketches."
    )

    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        dest="directory",
        help="Directory containing labelled folders of UI element sketches",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_folder",
        help="Output folder of cropped images",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        required=True,
        dest="limit",
        help="Number of dataset images to generate",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        dest="timeout",
        default=10,
        help="Subprocess timeout limit",
    )
    parser.add_argument(
        "-s",
        "--test-split",
        default=0.2,
        type=float,
        dest="test_split",
        help="Test split percentage (default: 0.2)",
    )

    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument(
        "--datagen-only",
        dest="datagen_only",
        action="store_true",
        help="Generate images and annotations without csv/tf shards",
    )
    parser_group.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        help="Generate only evaluation data without train/test",
    )
    parser_group.add_argument(
        "--conversion-only",
        dest="conversion_only",
        action="store_true",
        help="Converts data to csv & tf shards without train/test",
    )

    parser.set_defaults(eval_only=False)

    args = parser.parse_args()

    eval_only = args.eval_only
    if eval_only:
        print("** Evaluation data generation mode **")

    datagen_only = args.datagen_only
    if datagen_only:
        print("** Data generation mode **")

    conversion_only = args.conversion_only
    if datagen_only:
        print("** Data generation mode **")

    directory = args.directory
    directory = directory.strip(os.sep)
    print(f"Directory with UI element sketches: {directory}")

    output_folder = args.output_folder
    output_folder = output_folder.strip(os.sep)
    print(f"Output folder: {output_folder}")

    limit = args.limit
    print(f"{limit} images will be generated")

    timeout = args.timeout

    # Create Datagen instance
    uisketch_datagen = UISketch_Synthetic_Datagen(directory, output_folder)

    if not conversion_only:
        # Generate dataset
        uisketch_datagen.generate(limit=limit, timeout=timeout)

    if not datagen_only:
        if not eval_only:
            test_split = args.test_split
            print(f"Test split: {test_split}")

            xml_to_csv = XML_to_CSV(
                uisketch_datagen.annotations_folder,
                uisketch_datagen.data_folder,
            )

            xml_to_csv.convert(test_split=test_split)

            generate_tf_records = Generate_TF_Records(
                uisketch_datagen.image_folder,
                uisketch_datagen.data_folder,
                uisketch_datagen.labels,
            )

            generate_tf_records.generate_records(
                xml_to_csv.training_labels_csv, record_name="train.record"
            )
            generate_tf_records.generate_records(
                xml_to_csv.test_labels_csv, record_name="test.record"
            )
        else:
            xml_to_csv = XML_to_CSV(
                uisketch_datagen.annotations_folder,
                uisketch_datagen.data_folder,
            )

            XML_DF = xml_to_csv.xml_to_csv()

            XML_DF.to_csv(xml_to_csv.uisketch_labels_csv, index=None)

            generate_tf_records = Generate_TF_Records(
                uisketch_datagen.image_folder,
                uisketch_datagen.data_folder,
                uisketch_datagen.labels,
            )

            generate_tf_records.generate_records(
                xml_to_csv.uisketch_labels_csv, record_name="eval.record"
            )
