import os

import joblib
import pandas as pd

from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.models.model_creation import ModelCreator


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        models = self.config.params

        model_config = models.model
        model_creator = ModelCreator()
        model = model_creator.create_model(model_config)
        model.fit(train_x, train_y)
        joblib.dump(
            model, os.path.join(self.config.root_dir, models.model.name + ".joblib")
        )
