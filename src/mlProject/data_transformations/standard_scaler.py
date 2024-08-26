import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlProject import logger


class StandardScalerSK:

    def __init__(self, config, inference):
        self.config = config
        self.inference = inference
        for transformation in config.transformation_config:
            if transformation.get("name") == "StandardScalerSK":
                self.with_mean = transformation.get("with_mean")

    def transform_data(self, data):
        if self.inference:
            scaler = joblib.load(os.path.join(self.config.root_dir, "scaler.pkl"))
            data = scaler.transform(data)
            data.to_csv(os.path.join(self.config.root_dir, "data.csv"), index=False)
        else:
            # Split the data into training and test sets. (0.75, 0.25) split.
            scaler = StandardScaler()
            train = pd.read_csv(os.path.join(self.config.root_dir, "train.csv"))
            test = pd.read_csv(os.path.join(self.config.root_dir, "test.csv"))
            train_target = train[self.config.target.name]
            test_target = test[self.config.target.name]
            train_scaled = pd.DataFrame(
                scaler.fit_transform(train.drop(columns=[self.config.target.name])),
                columns=train.drop(columns=[self.config.target.name]).columns,
            )
            test_scaled = pd.DataFrame(
                scaler.transform(test.drop(columns=[self.config.target.name])),
                columns=test.drop(columns=[self.config.target.name]).columns,
            )

            # Add the "death" column back
            train = pd.concat([train_scaled, train_target], axis=1)
            test = pd.concat([test_scaled, test_target], axis=1)
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
            joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.pkl"))

            logger.info("Scaled the data")
