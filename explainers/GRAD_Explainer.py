from math import sqrt
import random
import numpy as np
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from tqdm import tqdm

from explainers.BaseExplainer import BaseExplainer
from utils.dataset_util.data_utils import index_edge
from utils.wandb_logger import WandbLogger

"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class GRADExplainer(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type, epochs=30, lr=0.003, reg_coefs=(0.03, 0.01)):
        super().__init__(model_to_explain, graphs, features, task, model_type, loss_type)
        #  init logger
        self.config = {
            'name': 'GRAD'
        }

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.train()

        feats = self.features[index].detach()
        graph = self.graphs[index].detach()
        # label = self.
        # Remove self-loops
        graph = graph[:, (graph[0] != graph[1])]

        # graph = torch.tensor(graph, requires_grad=True)
        # graph = graph.type(torch.float)
        # graph = torch.tensor(graph, requires_grad=True)
        expl_graph_weights = torch.ones(graph.size(1), requires_grad=True)
        original_pred = self.model_to_explain(feats, graph, edge_weights=expl_graph_weights)
        original_pred.backward()

        dg = expl_graph_weights.grad
        # print(dg)

        # Retrieve final explanation
        mask = torch.sigmoid(dg)
        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, dg.size(0)):  # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights

    def serialize_configuration(self):
        return 'GRAD_Explainer'
