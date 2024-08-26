# from mlProject.models import ElasticNet
from sklearn.linear_model import ElasticNet

RANDOM_STATE = 42


class ModelCreator:

    def __init__(self, model_config):
        self.model_config = model_config

    def create_model(self, model_config):
        model_type = model_config["model_type"]
        if model_type == "ElasticNet":
            self.alpha = self.model_config.alpha
            self.l1_ratio = self.model_config.l1_ratio
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=RANDOM_STATE,
            )
        return model
