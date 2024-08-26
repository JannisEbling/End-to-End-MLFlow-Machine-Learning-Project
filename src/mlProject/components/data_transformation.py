import os

import pandas as pd
from sklearn.model_selection import train_test_split

from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, inference=False):
        self.config = config
        self.inference = inference

    def apply_transformations(self):
        data = pd.read_csv(self.config.data_path)
        for transformation in self.config.transformation_config:
            # Create an instance of the class
            class_name = transformation.name
            if class_name in globals():
                Transformator = globals()[class_name](
                    config=self.config, inference=self.inference
                )
                Transformator.transform_data(data=data)
            else:
                raise NameError(f"Class {class_name} is not defined.")

        return data
