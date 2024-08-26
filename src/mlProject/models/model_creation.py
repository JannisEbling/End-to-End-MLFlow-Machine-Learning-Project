# from mlProject.models import ElasticNet
from sklearn.linear_model import RidgeClassifier

RANDOM_STATE = 42


class ModelCreator:

    def __init__(self):
        pass

    def create_model(self, model_config):
        self.model_config = model_config
        model_type = self.model_config.name
        if model_type == "RidgeClassifier":
            self.alpha = self.model_config.alpha
            model = RidgeClassifier(
                alpha=self.alpha,
                random_state=RANDOM_STATE,
            )
        return model
