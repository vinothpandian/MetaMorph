"""Split given pandas dataframe into train/test"""

import sys

import numpy as np
import pandas as pd


def split_labels(xml_df, test_split=0.2):
    """Split given pandas dataframe into train/test

    Arguments:
        xml_df {DataFrame} -- Pandas dataframe object

    Keyword Arguments:
        test_split {float} -- test dataset ratio (default: {0.2})

    Returns:
        tuple -- (train, test) pandas dataframe objects in a tuple
    """

    test_split = round(test_split, 1)
    train_split = round(1 - test_split, 1)

    if train_split <= 0:
        print("Train/Test split not valid.")
        sys.exit()

    grouped = xml_df.groupby("filename")
    grouped_list = [grouped.get_group(x) for x in grouped.groups]

    dataset_size = len(grouped_list)
    test_dataset_size = int(dataset_size * test_split)
    training_dataset_size = dataset_size - test_dataset_size

    print(80 * "#")
    print(f"Dataset size : {dataset_size}")
    print(f"Training dataset size : {training_dataset_size}")
    print(f"Test dataset size : {test_dataset_size}")
    print(80 * "#")

    train_index = np.random.choice(len(grouped_list), size=training_dataset_size, replace=False)

    test_index = np.setdiff1d(list(range(dataset_size)), train_index)

    train = pd.concat([grouped_list[i] for i in train_index])
    test = pd.concat([grouped_list[i] for i in test_index])

    return (train, test)
