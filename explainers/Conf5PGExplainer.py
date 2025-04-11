import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from tqdm import tqdm
import random
import numpy as np
import time
from explainers.BaseExplainer import BaseExplainer
from utils.dataset_util.data_utils import index_edge


class Conf5PGExplainer(BaseExplainer):
    """
    An implementation of the PGExplainer model with confidence model.
    This model is used to explain the predictions of a GNN model.
    The confidence score is on the edge level.
    """

    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type, epochs=100, lr=0.005, temp=(5.0, 1.0),
                 reg_coefs=(0.0003, 0.3), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task, model_type, loss_type)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias

        self.config = {
            'epochs': self.epochs,
            'lr': self.lr,
            'temp': self.temp,
            'reg_coefs': self.reg_coefs,
            'sample_bias': self.sample_bias,
            'task': self.task,
            'type': self.model_type,
            'loss_name': self.loss_name,
        }

        if self.task == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

    # @func_timer
    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.task == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.confidence_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(40, 4),
            nn.ReLU(),
            # nn.Sigmoid(),
        )

        self.confidence_model_part2 = nn.Sequential(
            nn.Linear(5, 1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        if indices is None:  # Consider all indices
            indices = range(0, self.graphs.size(0))

        # Create optimizer and temperature schedule
        self.optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        self.optimizer2 = Adam(list(self.confidence_model.parameters()) + list(self.confidence_model_part2.parameters()), lr=self.lr)
        self.temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # self.train(indices=indices)
        self.confidence_auc_score = []

    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()
        self.confidence_model.train()
        self.confidence_model_part2.train()

        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.task == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, 2 * self.epochs)):
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.task == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                confidence_embedding = self.confidence_model(input_expl).squeeze()
                # confidence_score = torch.sigmoid(confidence_score)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze(dim=0)
                confidence_embedding = torch.cat([confidence_embedding, mask], 1)

                confidence_score = self.confidence_model_part2(confidence_embedding)

                conf_mask = mask * confidence_score + (1 - confidence_score) * torch.randn(confidence_score.size())
                conf_mask = torch.sigmoid(conf_mask)
                masked_pred = self.model_to_explain(feats, graph, edge_weights=conf_mask)
                original_pred = self.model_to_explain(feats, graph)

                if self.task == 'node':  # we only care for the prediction of the node
                    masked_pred = masked_pred[n]
                    original_pred = original_pred[n]

                id_loss = self.loss(masked_pred, original_pred, conf_mask, self.reg_coefs)
                loss += id_loss

                conf_loss = self.confidence_loss(confidence_score, mask, self.ground_truth[n])
                loss += conf_loss
                # loss += nll_loss(mask, t)
            # print(e, loss)
            loss = loss.to(torch.float32)
            loss.backward()
            if e % 2 == 0:
                self.optimizer.step()
            else:
                self.optimizer2.step()

    def train_1_epoch(self, e, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()
        self.confidence_model.train()
        self.confidence_model_part2.train()

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.task == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop

        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        t = self.temp_schedule(e)

        expl_loss = torch.FloatTensor([0.0]).detach()
        confidence_loss = torch.FloatTensor([0.0]).detach()
        ib_loss_data = [0.0, 0.0, 0.0]  # [cce_loss, size_loss, entropy_loss]

        for i in range(2):
            loss = torch.FloatTensor([0.0]).detach()
            for n in indices:
                n = int(n)
                if self.task == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                confidence_embedding = self.confidence_model(input_expl).squeeze()
                # confidence_score = torch.sigmoid(confidence_score)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                confidence_embedding = torch.cat([confidence_embedding, mask.unsqueeze(dim=1)], 1)

                confidence_score = self.confidence_model_part2(confidence_embedding).squeeze()
                conf_mask = mask * confidence_score + (1 - confidence_score) * torch.randn(confidence_score.size())
                conf_mask = torch.sigmoid(conf_mask)
                masked_pred = self.model_to_explain(feats, graph, edge_weights=conf_mask)
                original_pred = self.model_to_explain(feats, graph)

                if self.task == 'node':  # we only care for the prediction of the node
                    masked_pred = masked_pred[n]
                    original_pred = original_pred[n]

                id_loss = self.loss(masked_pred, original_pred, conf_mask, self.reg_coefs)
                loss += id_loss

                conf_loss = self.confidence_loss(confidence_score, mask, self.ground_truth[n])
                loss += conf_loss

                if i % 2 == 0:
                    expl_loss += id_loss
                    ib_loss_data[0] += self.loss_data['loss/cce_loss']
                    ib_loss_data[1] += self.loss_data['loss/size_loss']
                    ib_loss_data[2] += self.loss_data['loss/mask_ent_loss']
                else:
                    confidence_loss += conf_loss
                # loss += nll_loss(mask, t)
                # print(e, loss)
            loss = loss.to(torch.float32)
            # train_1_epoch_loss = loss.detach().numpy().tolist()
            loss.backward()
            if i % 2 == 0:
                expl_loss = expl_loss.detach().numpy().tolist()
                self.optimizer.step()
            else:
                confidence_loss = confidence_loss.detach().numpy().tolist()
                self.optimizer2.step()
        self.ib_loss_data = [i / len(indices) for i in ib_loss_data]
        return expl_loss, confidence_loss

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        self.explainer_model.eval()
        self.confidence_model.eval()
        self.confidence_model_part2.eval()

        index = int(index)
        if self.task == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        confidence = self.confidence_model(input_expl).squeeze()
        confidence = self.confidence_model_part2(torch.cat([confidence, mask.unsqueeze(dim=1)], 1)).squeeze()
        # confidence = torch.sigmoid(confidence)

        # mask *= confidence
        # mask = torch.sigmoid(mask)

        expl_graph_weights = torch.zeros(graph.size(1))  # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        ca_score = [index, confidence.detach().numpy().tolist(), mask.detach().numpy().tolist()]
        self.confidence_auc_score.append(ca_score)
        # print(self.confidence_auc_score)
        return graph, expl_graph_weights

    def explain_with_noisy(self, index, noisy_mod):
        index = int(index)

        def produce_noisy_graph_1(graph, epsilon=0.5):
            noisy_graph = [[], []]
            for i in range(graph.size(1)):
                random.seed(42)
                t = random.random()
                if t >= epsilon:
                    noisy_graph[0].append(graph[0][i])
                    noisy_graph[1].append(graph[1][i])
                else:
                    pass

            if not noisy_graph[0]:
                noisy_graph = [[0, 1], [1, 0]]

            return torch.tensor(noisy_graph)

        def produce_noisy_graph_add_1_edge(n_graph, epsilon=0):
            epsilon *= 100
            epsilon = int(epsilon)
            random.seed(epsilon)
            nodes = list(set(n_graph.numpy().flatten().tolist()))
            t1 = random.choice(nodes)
            t2 = random.choice(nodes)
            edge = torch.tensor([[t1], [t2]])
            n_graph = torch.concat([n_graph, edge], dim=1)
            return n_graph

        def produce_noisy_graph_add_embedding_noisy(n_graph_ebd, epsilon=0):
            epsilon *= 0.001
            random.seed(int(epsilon*10))
            noisy_ebd = n_graph_ebd.clone().detach()
            noisy = torch.mean(n_graph_ebd) * epsilon
            noisy_ebd = torch.add(noisy_ebd, noisy)
            return torch.tensor(noisy_ebd)

        def produce_noisy_graph_add_embedding_noisy_node_level(n_graph_ebd, epsilon=0):
            epsilon *= 2
            random.seed(int(epsilon*10))
            noisy_ebd = n_graph_ebd.clone().detach()
            noisy_ebd = torch.add(noisy_ebd, epsilon)
            return torch.tensor(noisy_ebd)

        def produce_noisy_graph_reduce_motif_weight(n_graph, epsilon=0):
            edge_weight = [1.0 for i in range(n_graph.size(1))]
            for i in range(n_graph.size(1)):
                if self.ground_truth[index][i] == 1:
                    edge_weight[i] = 1 - epsilon
            return torch.tensor(edge_weight)

        if self.task == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        ca_score = [index, []]
        graphs = []
        expls = []
        original_graph = graph.clone().detach()

        for epsilon in [i * 0.01 for i in range(0, 100)]:
            if noisy_mod == 1:
                graph = produce_noisy_graph_add_1_edge(original_graph, epsilon=epsilon)
            elif noisy_mod == 2:
                graph = produce_noisy_graph_1(original_graph, epsilon=epsilon)
            elif noisy_mod == 3:
                embeds = produce_noisy_graph_add_embedding_noisy(embeds, epsilon=epsilon)
            elif noisy_mod == 4:
                embeds = produce_noisy_graph_add_embedding_noisy_node_level(embeds, epsilon=epsilon)
            elif noisy_mod == 5:
                edge_weight = produce_noisy_graph_reduce_motif_weight(graph, epsilon=epsilon)
                embeds = self.model_to_explain.embedding(feats, graph, edge_weights=edge_weight).detach()
            # graph = torch.tensor(graph)
            # Use explainer mlp to get an explanation
            input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
            if noisy_mod == 5:
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, training=False).squeeze() * edge_weight
            else:
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, training=False).squeeze()

            confidence = self.confidence_model(input_expl).squeeze()
            confidence = self.confidence_model_part2(torch.cat([confidence, mask.unsqueeze(dim=1)], 1)).squeeze()
            # confidence = self.confidence_model(input_expl).squeeze()
            # confidence = torch.sigmoid(confidence)

            # mask *= confidence
            # mask = torch.sigmoid(mask)

            expl_graph_weights = torch.zeros(graph.size(1))  # Combine with original graph
            for i in range(0, mask.size(0)):
                pair = graph.T[i]
                t = index_edge(graph, pair)
                expl_graph_weights[t] = mask[i]

            graphs.append(graph.clone().detach())
            expls.append(expl_graph_weights)
            ca_score[1].append([epsilon, confidence.detach().numpy().tolist(), graph.detach().numpy().tolist(), mask.detach().numpy().tolist()])

            # del graph, input_expl, sampling_weights, mask, confidence, expl_graph_weights
        self.confidence_auc_score.append(ca_score)
        # print(self.confidence_auc_score)
        return graphs, expls

    def serialize_configuration(self):
        return 'lr_' + str(self.lr) + '_epochs_' + str(self.epochs) + '_reg_coefs_' + str(self.reg_coefs[0]) + '_' + \
               str(self.reg_coefs[1]) + '_sample_bias_' + str(self.sample_bias) + '_temp_' + str(self.temp[0]) + \
               str(self.temp[1]) + \
               '_loss_' + self.loss_name + 'conf_loss_weight_' + str(self.conf_loss_weight)
