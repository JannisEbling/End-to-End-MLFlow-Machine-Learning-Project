import os

import logger
import pandas as pd
from sklearn.model_selection import train_test_split


class TrainTestSplitter:

    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.split_ratio = self.config.datatransformation

    def transform_data(self):

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(self.data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
