import shutil

import torch
import os

from explainers.GNNExplainer import GNNExplainer
from explainers.PGExplainer import PGExplainer
from explainers.GRAD_Explainer import GRADExplainer
from explainers.GAT_Explainer import GATExplainer
from utils.wandb_logger import WandbLogger
from explainers.Conf5PGExplainer import Conf5PGExplainer
from explainers.Conf6PGExplainer import Conf6PGExplainer

from explainers.PGExplainer_deep_ensemble import PGExplainer_deep_ensemble
from explainers.PGExplainer_bootstrap_ensemble import PGExplainer_bootstrap_ensemble

class ExplainerSelector:
    def __init__(self, explainer_name, model_name, dataset_name, model_to_explain,
                 loss_type, graphs, features):
        self.explainer_name = explainer_name

        self.model_name = model_name
        if model_name in ['GraphGCN', 'GAT']:
            self.task = 'graph'
        else:
            self.task = 'node'

        self.dataset_name = dataset_name
        if dataset_name in ['bareg1', 'bareg2', 'bareg3', 'crippen', 'triangles', 'triangles_small']:
            self.model_type = 'reg'
        else:
            self.model_type = 'cls'

        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features

        self.loss_type = loss_type
        self.explainer = self.select_explainer_model()

    def select_explainer_model(self):
        if self.explainer_name == 'GNNE':
            return GNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'PGE':
            return PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                               self.loss_type)
        elif self.explainer_name == 'GRAD':
            return GRADExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                 self.loss_type)
        elif self.explainer_name == 'GAT':
            return GATExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'CONF5PGE':
            return Conf5PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'CONF6PGE':
            return Conf6PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'PGEDE':
            return PGExplainer_deep_ensemble(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                               self.loss_type)
        elif self.explainer_name == 'PGEBE':
            return PGExplainer_bootstrap_ensemble(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                               self.loss_type)
        else:
            assert 0
