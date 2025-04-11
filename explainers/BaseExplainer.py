from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.contrastive_loss import ContrastiveLoss


class BaseExplainer(ABC):
    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type):
        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features
        self.task = task
        self.model_type = model_type  # reg or cls
        self.loss_type = loss_type  # ib(information bottleneck), inpl(inner product loss) or ctl(contrastive loss)
        self.Contrastive_loss = ContrastiveLoss()
        self.beta = 1.0
        self.conf_loss_weight = 1.0
        if loss_type == 'ib':
            if self.model_type == 'cls':
                self.loss_name = 'ibcls'
            elif self.model_type == 'reg':
                self.loss_name = 'ibreg'
        elif loss_type == 'inpl':
            self.loss_name = 'inpl'
        elif loss_type == 'ctl':
            self.loss_name = 'ctl'
        elif loss_type == 'ibci':
            self.loss_name = 'ibci'
        elif loss_type in ['ib_llm']:
            self.loss_name = loss_type
        elif loss_type.startswith('ibconf'):
            self.loss_name = loss_type
        else:
            assert 0
        self.loss_data = {}

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

    @abstractmethod
    def serialize_configuration(self):
        """
        Main method for output configuration
        :return: a text sequence
        """
        pass

    def build_neighbors_sim_score(self, indices):
        self.neighbor_pairs = {}
        self.neighbor_matrix = {idx: {} for idx in indices}
        # for index in range(len(self.graphs)):
        for index in indices:
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            with torch.no_grad():
                # original_pred1 = self.model_to_explain(feats, graph)
                # pred_label1 = original_pred1.detach()
                original_ebd = self.model_to_explain.build_graph_vector(feats, graph).detach().numpy()
            self.neighbor_pairs[index] = [original_ebd, -1]
        for _, key in enumerate(self.neighbor_pairs):
            for _, key2 in enumerate(self.neighbor_pairs):
                if key == key2:
                    self.neighbor_matrix[key][key2] = 1.0
                    continue
                if key2 < key:
                    self.neighbor_matrix[key][key2] = self.neighbor_matrix[key2][key]
                    continue
                cos_sim = cosine_similarity(self.neighbor_pairs[key][0], self.neighbor_pairs[key2][0])
                self.neighbor_matrix[key][key2] = cos_sim.tolist()[0][0]
        return

    def sample_index(self, index_, if_reverse):
        tmp = self.neighbor_matrix[index_]
        indices = []
        weight_list = []
        for _, key in enumerate(tmp):
            indices.append(key)
            if if_reverse:
                weight_list.append(1.0 - tmp[key])
            else:
                weight_list.append(tmp[key])
        while True:
            target = random.choices(indices, weights=weight_list, k=1)[0]
            if target != index_:
                break
        return target

    def loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        if self.loss_type == 'inpl':
            return self.inner_product_loss(masked_pred, original_pred)
        elif self.loss_type == 'ctl':
            return self.contrastive_loss(masked_pred, original_pred, edge_mask, reg_coefs)
        elif self.loss_type == 'ib_llm':
            return self.ib_llm_loss(masked_pred, original_pred, edge_mask, reg_coefs)
        elif self.loss_type in ['ib'] or self.loss_type.startswith('ibconf'):
            return self._loss(masked_pred, original_pred, edge_mask, reg_coefs)
        else:
            assert 0

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15  # 1e-4

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if self.model_type == 'cls':
            if self.task == 'node':
                pred_label = original_pred.detach()
                cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
                self.loss_data = {
                    'loss/cce_loss': cce_loss.detach().numpy().tolist(),
                    'loss/size_loss': size_loss.detach().numpy().tolist(),
                    'loss/mask_ent_loss': mask_ent_loss.detach().numpy().tolist(),
                }
                return cce_loss + size_loss + mask_ent_loss
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data['loss/cce_loss'] = cce_loss.detach().numpy().tolist()
            self.loss_data['loss/size_loss'] = size_loss.detach().numpy().tolist()
            self.loss_data['loss/mask_ent_loss'] = mask_ent_loss.detach().numpy().tolist()
            return cce_loss + size_loss + mask_ent_loss
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy().tolist(),
                'loss/size_loss': size_loss.detach().numpy().tolist(),
                'loss/mask_ent_loss': mask_ent_loss.detach().numpy().tolist(),
            }
            return mse_loss + size_loss + mask_ent_loss
        else:
            assert 0

    def ib_llm_loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15  # 1e-4

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data['loss/cce_loss'] = cce_loss.detach().numpy().tolist()
            self.loss_data['loss/size_loss'] = size_loss.detach().numpy().tolist()
            self.loss_data['loss/mask_ent_loss'] = mask_ent_loss.detach().numpy().tolist()
            return self.llm_score * cce_loss + size_loss + mask_ent_loss
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy().tolist(),
                'loss/size_loss': size_loss.detach().numpy().tolist(),
                'loss/mask_ent_loss': mask_ent_loss.detach().numpy().tolist(),
            }
            return self.llm_score * mse_loss + size_loss + mask_ent_loss
        else:
            assert 0

    def ib_confidence_loss(self, pred, gt, loss_type='nll'):
        return 0.0
        if loss_type == 'nll':
            m = nn.LogSoftmax(dim=1)
            loss = nn.NLLLoss()
            input = torch.randn(3, 5, requires_grad=True)
            target = torch.tensor([1, 0, 4])
            output = loss(m(input), target)
            return output
        else:
            assert 0

    def gib_without_mse(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        # mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        inpl = self.inner_product_loss(mask, None)
        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data = {
                'loss/cce_loss': cce_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return size_loss + inpl
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return size_loss + inpl
        else:
            assert 0

    def contrastive_loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15
        beta = torch.tensor(self.beta, requires_grad=False)
        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        # mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        inpl = self.inner_product_loss(mask, None)
        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data = {
                'loss/cce_loss': cce_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return beta * cce_loss + size_loss + inpl
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return beta * mse_loss + size_loss + inpl
        else:
            assert 0

    def inner_cce_loss(self, masked_pred, original_pred):
        inner_product_loss = -torch.log(torch.sigmoid(masked_pred) * torch.sigmoid(original_pred)).sum()
        self.loss_data = {
            'loss/inner_product_loss': inner_product_loss.detach().numpy(),
        }
        return inner_product_loss

    def nll_loss(self, idx, graph_, expl_):
        y_true = []
        y_pred = expl_
        graph = graph_.detach().numpy()
        expl = expl_
        ground_truth = self.ground_truth[idx]
        if self.task == 'node':
            for i in range(0, expl.shape[0]):
                ground_truth_graph = self.graphs.numpy()
                ground_truth = self.ground_truth
                pair = graph.T[i]

                a = (ground_truth_graph.T == pair)
                a = a.all(axis=1)
                idx_edge = np.where(a)
                idx_edge = idx_edge[0]
                idx_edge_rev = np.where((ground_truth_graph.T == [pair[1], pair[0]]).all(axis=1))[0]

                # If any of the edges is in the ground truth set, the edge should be in the explanation
                gt = ground_truth[idx_edge] + ground_truth[idx_edge_rev]
                if gt == 0:
                    y_true.append(0)
                else:
                    y_true.append(1)
                # y_pred.append(expl[i])
        else:
            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                # Retrieve predictions and ground truth
                y_pred.append(expl[edge_idx])
                y_true.append(ground_truth[edge_idx])
        y_true = torch.tensor(y_true, dtype=torch.float)
        # len_y_pred = y_pred.shape[0]
        # y_pred_2 = torch.ones(len_y_pred)
        # y_pred_2 = y_pred_2 - y_pred
        # y_pred = torch.unsqueeze(y_pred, 1)
        # y_pred_2 = torch.unsqueeze(y_pred_2, 1)
        # y_pred = torch.cat((y_pred_2, y_pred), 1)
        m = torch.nn.Sigmoid()
        y_pred = m(y_pred)
        loss = torch.nn.BCELoss()
        nll_loss = loss(y_pred, y_true)
        return nll_loss

    def inner_product_loss(self, masked_pred, original_pred):
        inpl = masked_pred * masked_pred
        inpl = torch.sigmoid(inpl)
        inpl = -torch.log(inpl)
        inner_product_loss = inpl.sum()
        self.loss_data = {
            'loss/inner_product_loss': inner_product_loss.detach().numpy(),
        }
        return inner_product_loss

    def cross_inner_product_loss(self, masked_pred, original_pred):
        inpl = masked_pred * original_pred
        inpl = torch.sigmoid(inpl)
        inpl = -torch.log(inpl)
        inner_product_loss = inpl.sum()
        self.loss_data['loss/inner_product_loss'] = inner_product_loss.detach().numpy()
        return inner_product_loss

    def confidence_loss(self, confidence_score, mask, gt):
        gt = gt.type(torch.FloatTensor)
        # confidence_loss = torch.nn.functional.mse_loss(confidence_score, gt)
        if self.loss_name == 'ibconf1':
            confidence_loss = self.conf_loss_weight * (torch.sigmoid(confidence_score) *
                                                       torch.sigmoid(torch.abs(gt - mask))).sum()
        elif self.loss_name == 'ibconf1_2':
            confidence_loss = self.conf_loss_weight * (torch.sigmoid(confidence_score) *
                                                       nn.functional.cross_entropy(gt, mask)).sum()
        elif self.loss_name == 'ibconf1_3':
            confidence_loss = self.conf_loss_weight * (torch.sigmoid(confidence_score) *
                                                       torch.abs(gt - mask)).sum()
        elif self.loss_name == 'ibconf2':
            tmp = 1 - mask
            mask = mask.unsqueeze(dim=1)
            tmp = tmp.unsqueeze(dim=1)
            mask2 = torch.concat((mask, tmp), dim=1)
            mask_pred = torch.argmax(mask2, dim=1)
            # mask_pred = mask_pred.view(-1, 1)
            # mask_pred = 1 - mask_pred
            mask_3 = torch.logical_xor(mask_pred, gt)
            mask_3 = mask_3.type(torch.FloatTensor)  # [1, 0, 1, 0...]
            # mask_3 = mask_3.view(-1, 1)
            # error mask3 = 1 - mask_3
            mask_3 = 2 * mask_3 - 1
            confidence_loss = self.conf_loss_weight * (
                    mask_3 * torch.sigmoid(confidence_score) * torch.abs(gt - mask.squeeze())).sum()
        elif self.loss_name == 'ibconf2_2':
            tmp = 1 - mask
            mask = mask.unsqueeze(dim=1)
            tmp = tmp.unsqueeze(dim=1)
            mask2 = torch.concat((mask, tmp), dim=1)
            mask_pred = torch.argmax(mask2, dim=1)
            # mask_pred = mask_pred.view(-1, 1)
            # mask_pred = 1 - mask_pred
            mask_3 = torch.logical_xor(mask_pred, gt)
            mask_3 = mask_3.type(torch.FloatTensor)  # [1, 0, 1, 0...]
            # mask_3 = mask_3.view(-1, 1)
            # error mask3 = 1 - mask_3
            mask_3 = 2 * mask_3 - 1
            confidence_loss = self.conf_loss_weight * (
                    mask_3 * torch.sigmoid(confidence_score) * nn.functional.cross_entropy(gt,
                                                                                           mask.squeeze())).sum()
        elif self.loss_name == 'ibconf2_3':
            tmp = 1 - mask
            mask = mask.unsqueeze(dim=1)
            tmp = tmp.unsqueeze(dim=1)
            mask2 = torch.concat((mask, tmp), dim=1)
            mask_pred = torch.argmax(mask2, dim=1)
            # mask_pred = mask_pred.view(-1, 1)
            # mask_pred = 1 - mask_pred
            mask_3 = torch.logical_xor(mask_pred, gt)
            mask_3 = mask_3.type(torch.FloatTensor)  # [1, 0, 1, 0...]
            # mask_3 = mask_3.view(-1, 1)
            # error mask3 = 1 - mask_3
            mask_3 = 2 * mask_3 - 1
            confidence_loss = self.conf_loss_weight * (
                    mask_3 * torch.sigmoid(confidence_score) * torch.sigmoid(torch.abs(gt - mask.squeeze()))).sum()
        elif self.loss_name == 'ibconf2_3_1':
            mask = mask.unsqueeze(dim=1)
            confidence_loss = self.conf_loss_weight * torch.sigmoid(torch.abs(gt - mask.squeeze())).sum()
        elif self.loss_name == 'ibconf3':
            tmp = 1 - mask
            mask = mask.unsqueeze(dim=1)
            tmp = tmp.unsqueeze(dim=1)
            mask2 = torch.concat((mask, tmp), dim=1)
            mask_pred = torch.argmax(mask2, dim=1)
            # mask_pred = mask_pred.view(-1, 1)
            # mask_pred = 1 - mask_pred
            mask_3 = torch.logical_xor(mask_pred, gt)
            mask_3 = mask_3.type(torch.FloatTensor)  # [1, 0, 1, 0...]
            # mask_3 = mask_3.view(-1, 1)
            # error mask3 = 1 - mask_3
            mask_3 = 2 * mask_3 - 1
            confidence_loss = self.conf_loss_weight * (
                    mask_3 * confidence_score * torch.abs(gt - mask.squeeze())).sum()
        elif self.loss_name == 'ibconf4':
            #  we design the confidence loss with collaborate loss here
            #  calculate collaboration_loss
            # todo
            tmp = 1 - mask
            mask = mask.unsqueeze(dim=1)
            tmp = tmp.unsqueeze(dim=1)
            mask2 = torch.concat((tmp, mask), dim=1)
            mask_pred = torch.argmax(mask2, dim=1)
            # mask_pred = mask_pred.view(-1, 1)
            # mask_pred = 1 - mask_pred
            mask_3 = torch.sum(mask_pred == gt) / len(gt)
            confidence_loss = self.conf_loss_weight * (confidence_score - mask_3) ** 2
        elif self.loss_name == 'ibconf5':
            confidence_loss = self.conf_loss_weight * (confidence_score *
                                                       torch.abs(gt - mask)).sum()
        else:
            assert 0

        self.loss_data['loss/confidence_loss'] = confidence_loss.detach().numpy(),

        # for edge in gt:
        #     edge_confidence = - (boolean(y^*==y) * |exp[i]-gt[i]|/(gt[i]+bias))
        #     pass
        # gt_confidence = avg(edge_confidences)

        return confidence_loss
