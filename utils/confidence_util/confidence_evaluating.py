import json
import math
import os

import numpy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch_geometric.data import Data, DataLoader
from utils.dataset_util.dataset_loaders import DatasetLoader
from utils.dataset_util.data_utils import to_torch_graph
from gnns.model_selector import ModelSelector
from explainers.explainer_selector import ExplainerSelector
from utils.wandb_logger import WandbLogger
from utils.confidence_util.confidence_utils import save_confidence_results, load_confidence_results
import torch_geometric as ptgeom
import pandas as pd
from evaluating_sub_1 import cor_matrix
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr


class ConfidenceEvaluator:
    def __init__(self, dataset_name, model_name, explainer_name, loss_type='ib',
                 seeds=None, force_run=True, save_confidence_scores=True, temps=None):
        # init gnn and explainer
        if seeds is None:
            self.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if temps is None:
            self.temps = [5.0, 1.0]

        self.n_split_folds = 10

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.explainer_name = explainer_name

        self.loss_type = loss_type
        self.seed = seeds
        self.force_run = force_run
        self.save_confidence_scores = save_confidence_scores
        self.logging = False
        if model_name == 'RegGCN':
            self.input_type = 'dense'
        else:
            if dataset_name in ['ba2-2motifs']:
                self.input_type = 'dense'
            self.input_type = 'sparse'

        if dataset_name in ['bareg1', 'bareg2', 'crippen', 'bareg3', 'triangles', 'triangles_small']:
            self.type = 'reg'
        else:
            self.type = 'cls'

        if dataset_name in ['syn1', 'syn2', 'syn3', 'syn4', ]:
            self.task = 'node'
        else:
            self.task = 'graph'

        self.dataset_loader = DatasetLoader(self.dataset_name, self.input_type, self.type)
        self.dataset_loader.task = self.task
        self.dataset_loader.load_dataset()
        self.dataset_loader.create_data_list()

        self.graphs, self.features = self.dataset_loader.graphs, self.dataset_loader.features
        self.labels, self.test_mask = self.dataset_loader.labels, self.dataset_loader.test_mask
        self.ground_truth = self.dataset_loader.edge_ground_truth

        self.model_manager = ModelSelector(model_name, dataset_name, load_pretrain=True)
        self.model_manager.model.type = self.type
        self.explainer_manager = ExplainerSelector(explainer_name, model_name, dataset_name, self.model_manager.model,
                                                   self.loss_type, self.graphs, self.features)

        self.explainer_manager.explainer.ground_truth = self.ground_truth

        wandb_log = False
        project_name = explainer_name + '_' + dataset_name + '_' + model_name
        self.explainer_manager.explainer.wandb_logger = WandbLogger(project_name,
                                                                    self.explainer_manager.explainer.config, wandb_log)
        self.explainer_manager.explainer.total_step = 0
        pass

    def explain(self):
        if self.dataset_name in ['crippen', 'benzene', 'flca', 'alca', 'bareg3']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(np.array(self.features))

        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)
        self.explainer_manager.explainer.ground_truth = [torch.tensor(i) for i in self.ground_truth]
        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()

        # Perform the evaluation 10 times
        auc_scores = []
        binary_auc_scores = []
        rdd_scores = []
        nll_scores = []

        self.dataset_loader.data_root_path = '/data/jiaxing/xai_conf_result/data/'
        if not os.path.exists(self.dataset_loader.data_root_path):
            self.dataset_loader.data_root_path = '/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/'
        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/logs/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())

        if os.path.exists(folder_path) and not self.force_run:
            print('already finished!')
            return self.show_results()

        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())

        scores = []
        ca_scores = []
        loss_dict = {}
        for turn_, s in enumerate(self.seeds):
            loss_dict[s] = {'expl_loss': [], 'conf_loss': [],'ib_loss_data': [],  'scores': []}
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)

            record_each_epoch = False
            if record_each_epoch:
                for e in tqdm(range(0, self.explainer_manager.explainer.epochs)):
                    expl_loss, conf_loss = self.explainer_manager.explainer.train_1_epoch(e, indices)
                    loss_dict[s]['expl_loss'].append([e, expl_loss])
                    loss_dict[s]['conf_loss'].append([e, conf_loss])
                    try:
                        loss_dict[s]['ib_loss_data'].append([e, self.explainer_manager.explainer.ib_loss_data])
                    except:
                        pass
                    explanations = []
                    for idx in tqdm(indices, disable=(not self.logging)):
                        assert len(self.dataset_loader.edge_ground_truth[idx]) == len(self.dataset_loader.graphs[idx].T)
                        graph, expl = self.explainer_manager.explainer.explain(idx)
                        ground_truth = self.dataset_loader.edge_ground_truth[idx]
                        assert len(graph.T) == len(expl)
                        if len(ground_truth) != len(expl):
                            # print('ground truth: ', ground_truth)
                            # print('explanation: ', expl)
                            print('missing edge: ', idx)
                        explanations.append((idx, graph, ground_truth, expl))
                        # explanations.append((idx, graph, expl))

                    y_true, y_pred = self.prepare_evaluate(explanations)
                    auc_score = self.evaluate_auc(y_true, y_pred)
                    y_true, y_pred = self.prepare_binary_evaluate(explanations)
                    auc_binary_score = self.evaluate_auc(y_true, y_pred)
                    loss_dict[s]['scores'].append([e, auc_score, auc_binary_score])
            else:
                if self.explainer_name == 'GNNE':
                    pass
                else:
                    self.explainer_manager.explainer.train(indices)
            explanations = []
            for idx in tqdm(indices, disable=(not self.logging)):
                assert len(self.dataset_loader.edge_ground_truth[idx]) == len(self.dataset_loader.graphs[idx].T)
                graph, expl = self.explainer_manager.explainer.explain(idx)
                ground_truth = self.dataset_loader.edge_ground_truth[idx]
                assert len(graph.T) == len(expl)
                if len(ground_truth) != len(expl):
                    # print('ground truth: ', ground_truth)
                    # print('explanation: ', expl)
                    # print('missing edge: ', idx)
                    pass
                explanations.append((idx, graph, ground_truth, expl))

            y_true, y_pred = self.prepare_evaluate(explanations)
            auc_score = self.evaluate_auc(y_true, y_pred)
            rdd_score = self.evaluate_rdd(y_true, y_pred)
            nll_score = self.evaluate_nll(y_true, y_pred)

            print("auc score:", auc_score, "rdd score: ", rdd_score)
            # y_true, y_pred = self.prepare_binary_evaluate(explanations)
            # auc_binary_score = self.evaluate_auc(y_true, y_pred)
            # print("auc score:", auc_score, "binary auc score:", auc_binary_score, "rdd score: ", rdd_score)

            scores.append([s, auc_score, rdd_score, nll_score])

            if self.save_confidence_scores:
                file_name = str(s) + '_results.json'
                save_confidence_results([s, auc_score, rdd_score, nll_score], explanations, save_path=folder_path, file_name=file_name)

                # loss_dict_path = os.path.join(folder_path, 'loss_log.json')
                # with open(loss_dict_path, 'w') as f:
                #     f.write(json.dumps(loss_dict))

            # auc_scores.append(auc_score)
            # binary_auc_scores.append(auc_binary_score)
            # rdd_scores.append(rdd_score)
            # nll_scores.append(nll_score)

            # ca_scores.append([str(s), self.explainer_manager.explainer.confidence_auc_score])

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(save_score_path):
            os.makedirs(save_score_path)

        with open(os.path.join(save_score_path, 'scores.json'), 'w') as f:
            f.write(json.dumps(scores))

        if False:
            ca_path = os.path.join(save_score_path, 'confidence_auc_score.json')
            if not os.path.exists(save_score_path):
                os.makedirs(save_score_path)
            with open(ca_path, 'w') as f:
                f.write(json.dumps(ca_scores))

        self.show_results()

        return

    def evaluate_with_noisy(self, noisy_mod=4):
        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()

        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())

        print('noisy mod: ', noisy_mod)
        for turn_, s in enumerate(self.seeds):
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)
            self.explainer_manager.explainer.train(indices)

            explanations = []
            print('Adding noisy...')
            for idx in tqdm(indices, disable=(not self.logging)):
                graph, expl = self.explainer_manager.explainer.explain_with_noisy(idx, noisy_mod)
                explanations.append((idx, graph, expl))

            auc = []
            for i in range(len(explanations)):
                tmp = []
                for j in range(len(explanations[i][1])):
                    expl = [[explanations[i][0], explanations[i][1][j], explanations[i][2][j]]]
                    y_true, y_pred = self.prepare_evaluate_exd_graph(expl)
                    auc_score = self.evaluate_auc(y_true, y_pred)

                    tmp.append(auc_score)
                auc.append(tmp)
            ca_scores = [str(s), auc, self.explainer_manager.explainer.confidence_auc_score]

            if True:
                ca_path = os.path.join(save_score_path, 'noisy_confidence_auc_score_'+str(noisy_mod)+'_0_seed_'+str(s)+'.json')
                if not os.path.exists(save_score_path):
                    os.makedirs(save_score_path)
                with open(ca_path, 'w') as f:
                    f.write(json.dumps(ca_scores))
                print('save noisy auc score at ', ca_path)
        return

    def prepare_evaluate(self, explanations):
        y_true = []
        y_pred = []
        # print(explanations[0][1][0])
        # print('total edges in graph: ', sum([len(i[1][0]) for i in explanations]))
        # print('total edges in original graph: ', sum([len(self.dataset_loader.graphs[i[0]]) for i in explanations]))
        # print('total edges in ground truth: ', sum([len(i[2]) for i in explanations]))
        # print('total edges in explanation: ', sum([len(i[3]) for i in explanations]))
        for idx, graph_, ground_truth_, expl_ in explanations:
            try:
                graph = graph_.detach().numpy()
            except:
                graph = np.array(graph_)
            ground_truth = ground_truth_
            try:
                expl = expl_.detach().numpy()
            except:
                expl = np.array(expl_)
            original_graph = self.dataset_loader.graphs[idx]
            # print('graph: ', graph)
            # print('length of graph: ', len(graph.T))
            # print('expl: ', expl)
            # print('length of expl: ', len(expl))
            # print('original_graph: ', original_graph)
            # print('length of original_graph: ', len(original_graph.T))
            # print('ground_truth: ', ground_truth)
            # print('length of ground_truth: ', len(ground_truth))

            ground_truth = self.dataset_loader.edge_ground_truth[idx]
            if self.task == 'node':
                assert 0
                for i in range(0, expl.shape[0]):
                    ground_truth_graph = self.dataset_loader.graphs.numpy()
                    ground_truth = self.dataset_loader.edge_ground_truth
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

                    y_pred.append(expl[i])
                continue
            for edge_idx in range(graph.T.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                # find the edge in the original graph
                original_graph_edge_idx = np.where((original_graph.numpy().T == edge_).all(axis=1))

                # print('original_graph_edge_idx: ', original_graph_edge_idx)
                try:
                    original_graph_edge_idx = original_graph_edge_idx[0][0]
                except:
                    # print('origin', original_graph)
                    # print('new', graph)

                    # print(edge_)
                    # print(original_graph_edge_idx)
                    continue
                    assert 0
                    original_graph_edge_idx = original_graph_edge_idx[0]
                # Retrieve predictions and ground truth
                y_pred.append(expl[edge_idx])
                try:
                    y_true.append(float(ground_truth[original_graph_edge_idx]))
                except:
                    print('original_graph_edge_idx: ', original_graph_edge_idx)
                    assert 0
        # print('length of y_true: ', len(y_true), 'length of y_pred: ', len(y_pred))
        # print('y_true: ', y_true)
        # print('y_pred: ', y_pred)
        return np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32)

    def prepare_evaluate_exd_graph(self, explanations):
        y_true = []
        y_pred = []
        for idx, graph_, expl_ in explanations:
            graph = graph_.detach().numpy()
            expl = expl_.detach().numpy()
            ground_truth = self.dataset_loader.edge_ground_truth[idx]

            expl_graph = [[src, tgt] for src, tgt in graph.T]
            gt_graph = [[src, tgt] for src, tgt in self.dataset_loader.graphs[idx].T]
            for i in range(len(expl_graph)):
                if expl_graph[i] in gt_graph:
                    mark = gt_graph.index(expl_graph[i])
                    if ground_truth[mark] == 1:
                        y_true.append(1)
                    elif ground_truth[mark] == 0:
                        y_true.append(0)
                    else:
                        assert 0
                else:
                    y_true.append(0)
                y_pred.append(expl[i])

        return y_true, y_pred

    def prepare_binary_evaluate(self, explanations, threshold=6):
        y_true = []
        y_pred = []
        for idx, graph_, expl_ in explanations:
            graph = graph_.detach().numpy()
            expl = expl_.detach().numpy()

            if self.task == 'node':
                threshold = 0
                ground_truth_graph = self.dataset_loader.graphs.numpy()
                ground_truth = self.dataset_loader.edge_ground_truth
                for i in range(0, expl.shape[0]):
                    pair = graph.T[i]
                    a = (ground_truth_graph.T == pair)
                    a = a.all(axis=1)
                    idx_edge = np.where(a)
                    idx_edge = idx_edge[0]
                    idx_edge_rev = np.where((ground_truth_graph.T == [pair[1], pair[0]]).all(axis=1))[0]
                    # If any of the edges is in the ground truth set, the edge should be in the explanation
                    # tmp = ground_truth[idx_edge].item() + ground_truth[idx_edge_rev].item()
                    if ground_truth[idx_edge].item() or ground_truth[idx_edge_rev].item():
                        threshold += 1
            else:
                ground_truth = self.dataset_loader.edge_ground_truth[idx]
                threshold = numpy.sum(ground_truth)
                # assert 0
            threshold = int(threshold)
            ind = np.argpartition(expl, -threshold)[-threshold:]
            topk = expl[ind]
            binary_expl = numpy.zeros(expl.shape[0])
            binary_expl[ind] = 1.0

            if self.task == 'node':
                ground_truth_graph = self.dataset_loader.graphs.numpy()
                ground_truth = self.dataset_loader.edge_ground_truth
                for i in range(0, expl.shape[0]):
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

                    y_pred.append(binary_expl[i])
                continue
            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                # Retrieve predictions and ground truth
                y_pred.append(binary_expl[edge_idx])
                y_true.append(ground_truth[edge_idx])
        return y_true, y_pred

    def evaluate_auc(self, y_true, y_pred):
        try:
            score = roc_auc_score(y_true, y_pred)
            return score
        except:
            if len(y_true) == 0:
                return 0.0
            elif y_pred[0] != y_pred[0]:
                return 0.0
            else:
                return 0.0

    def evaluate_rdd(self, ground_truth, explanation):
        """
        :param: explanation: edge weight vector
        :param: ground_truth: binary/float edge ground truth
        evaluate the reversed distribution distance from explanation to ground truth by f(x_{N})= 1 - root(x_{i}^{2})/N
        :return: float
        """
        assert len(explanation) == len(ground_truth)
        sum_d = 0.0
        for i in range(len(explanation)):
            sum_d += (explanation[i] - ground_truth[i]) ** 2
        return 1 - (sum_d / len(explanation)) ** 0.5

    def evaluate_nll(self, tgt, src):
        # inputs = []
        # for i in src:
        #     inputs.append([1.0 - i, i])
        # m = torch.nn.LogSoftmax(dim=1)
        # nll = torch.nn.NLLLoss()
        # res = nll(torch.tensor(inputs), torch.LongTensor(tgt))
        # res = res.detach().item()
        src1 = torch.tensor(src)
        tgt1 = torch.tensor(tgt)
        # m = torch.nn.Sigmoid()
        # y_pred = m(src1)
        y_pred = src1
        loss = torch.nn.BCELoss()
        nll_loss = loss(tgt1, y_pred)

        return nll_loss.item()

    def evaluate_brier_score(self, ground_truth, predictions):
        assert len(ground_truth) == len(predictions)
        brier_score = brier_score_loss(ground_truth, predictions)
        return brier_score

    def evaluate_ece_mce_score(self, ground_truth, predictions, n_bins=10):
        """
        We estimate the expected calibration error (ECE) by dividing the predictions into N bins and computing the
        absolute difference between the accuracy and confidence in each bin.
        :param ground_truth:
        :param predictions:
        :param n_bins:
        :return:
        """
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        ece = 0
        mce = 0
        total_count = len(predictions)

        for i in range(n_bins):
            bin_mask = bin_indices == i
            bin_count = np.sum(bin_mask)

            if bin_count == 0:
                continue

            accuracy = np.mean(ground_truth[bin_mask])
            confidence = np.mean(predictions[bin_mask])
            bin_error = np.abs(accuracy - confidence)

            ece += (bin_count / total_count) * bin_error
            mce = max(mce, bin_error)

        return ece, mce

    def set_experiments(self):
        run_confidence_interval = False
        if run_confidence_interval:
            self.evaluate_confident_interval()
            self.evaluating()

        return self.explain()

    def evaluating(self):
        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/intervals/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())

        alpha = 0.5
        ci_s = []
        for s in range(self.n_split_folds - 1):
            file_name = str(s) + '_results.json'
            data = load_confidence_results(folder_path, file_name)
            seed = data[0]
            auc_score, rdd_score, nll_score = data[1], data[2], data[3]
            explanations = data[4]
            scores = {}
            for expl in explanations:
                idx = expl[0]
                expl_weight = expl[1]
                if idx in scores.keys():
                    scores[idx].append(expl_weight)
                else:
                    scores[idx] = [expl_weight]

            confidence_interval = []
            for _, key in enumerate(scores):
                temp = scores[key]
                a_bound = np.quantile(temp, alpha / 2)
                b_bound = np.quantile(temp, 1 - (alpha / 2))
                confidence_interval.append(abs(a_bound-b_bound))

            ci = np.mean(confidence_interval)
            ci_s.append(ci)
        ci_mean = np.mean(ci_s)
        ci_std = np.std(ci_s)
        print(self.dataset_name, self.model_name, self.explainer_name)
        print('confidence interval, mean: ', ci_mean, ', std: ', ci_std)

        """
        [1, 1, 0]
        [0.9, 0.9, 0.1] -> [0.99, 0.99, 0.11] -> []
        [0.9, 0.49, 0.51] -> [0.99, 0.01, 0.99]
        
        """
        pass

    def evaluate_confident_interval(self):
        if self.dataset_name in ['crippen', 'benzene']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(self.features)

        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)
        self.explainer_manager.explainer.ground_truth = [torch.tensor(i) for i in self.ground_truth]
        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()
        test_indices = np.argwhere(self.dataset_loader.test_mask).squeeze()
        # Perform the evaluation 10 times
        # indices.extend(test_indices)
        indices = np.concatenate((indices, test_indices))
        auc_scores = []
        binary_auc_scores = []
        rdd_scores = []
        nll_scores = []

        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/intervals/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())

        if os.path.exists(folder_path) and not self.force_run:
            print('already finished!')
            return
            # self.dataset_loader.show_results(self.explainer_name, self.dataset_name, self.model_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        n_split_folds = self.n_split_folds
        np.random.shuffle(indices)
        if len(indices) % n_split_folds == 0:
            n_indices = np.split(indices, n_split_folds)
            test_indices = n_indices[-1]
            n_indices = n_indices[:-1]
        else:
            full_len = len(indices)
            red_len = len(indices) % n_split_folds
            re_indices = indices[-red_len:]
            indices = indices[:full_len-red_len]

            n_indices = np.split(indices, n_split_folds)
            test_indices = n_indices[-1]
            n_indices = n_indices[:-1]
            test_indices = np.append(test_indices, re_indices, 0)
        for s in range(len(n_indices)):
            print(f"Run excluding fold {s} in {len(n_indices)}")
            # Set all seeds needed for reproducibility
            indices = np.delete(n_indices.copy(), s, 0)
            indices = np.concatenate(indices)

            # continue

            self.explainer_manager.explainer.prepare(indices)

            explanations = []
            for idx in tqdm(test_indices, disable=(not self.logging)):
                graph, expl = self.explainer_manager.explainer.explain(idx)
                explanations.append((idx, graph, expl))

            y_true, y_pred = self.prepare_evaluate(explanations)
            auc_score = self.evaluate_auc(y_true, y_pred)
            rdd_score = self.evaluate_rdd(y_true, y_pred)
            nll_score = self.evaluate_nll(y_true, y_pred)

            y_true, y_pred = self.prepare_binary_evaluate(explanations)
            auc_binary_score = self.evaluate_auc(y_true, y_pred)

            if self.save_confidence_scores:
                file_name = str(s) + '_results.json'
                save_confidence_results([s, auc_score, rdd_score, nll_score], explanations, save_path=folder_path, file_name=file_name)

            auc_scores.append(auc_score)
            binary_auc_scores.append(auc_binary_score)
            rdd_scores.append(rdd_score)
            nll_scores.append(nll_score)

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        binary_auc = np.mean(binary_auc_scores)
        binary_auc_std = np.std(binary_auc_scores)
        rdd = np.mean(rdd_scores)
        rdd_std = np.std(rdd_scores)
        nll = np.mean(nll_scores)
        nll_std = np.std(nll_scores)

        print("auc score:", auc, auc_std, "binary auc score:", binary_auc, binary_auc_std, "rdd score: ", rdd, rdd_std)
        return auc, auc_std, rdd, rdd_std, nll, nll_std
        pass

    def evaluate_acc_nll_coefficient(self):
        if self.dataset_name in ['crippen']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(self.features)

        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)

        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()

        # Perform the evaluation 10 times
        auc_scores = []
        rdd_scores = []
        nll_scores = []

        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/logs/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())

        if not os.path.exists(save_score_path):
            os.makedirs(save_score_path)

        scores = {}
        for turn_, s in enumerate(self.seeds):
            self.explainer_manager.explainer.prepare(indices)
            scores[int(s)] = {}
            file_name = str(s) + '_results.json'
            data = load_confidence_results(save_path=folder_path, file_name=file_name)
            expl_data = data[4]
            expl_data = {i[0]: i[1] for i in expl_data}
            for idx in tqdm(indices, disable=(not self.logging)):
                explanations = []
                graph = ptgeom.utils.k_hop_subgraph(int(idx), 3, self.explainer_manager.explainer.graphs)[1]
                expl = expl_data[idx]
                graph = torch.tensor(graph).clone()
                expl = torch.tensor(expl).clone()
                explanations.append((idx, graph, expl))

                y_true, y_pred = self.prepare_evaluate(explanations)
                try:
                    auc_score = self.evaluate_auc(y_true, y_pred)
                except:
                    auc_score = -1
                    continue
                rdd_score = self.evaluate_rdd(y_true, y_pred)
                nll_score = self.evaluate_nll(y_true, y_pred)
                # print("auc score:", auc_score, "rdd score: ", rdd_score, 'nll score: ', nll_score)

                auc_scores.append(auc_score)
                rdd_scores.append(rdd_score)
                nll_scores.append(nll_score)

                scores[int(s)][idx.tolist()] = [auc_score, rdd_score, nll_score]

        with open(os.path.join(save_score_path, 'auc_nll_coefficient_scores.json'), 'w') as f:
            f.write(json.dumps(scores))

        data = {
            '$AUC$': auc_scores,
            '$LL$': nll_scores,
            '$RDD$': rdd_scores,
        }

        df = pd.DataFrame(data)
        df = df.iloc[1:]
        plt.rcParams.update({'font.size': 22})
        # with regression
        sns.pairplot(df, kind="reg")
        plt.savefig("correlation"+'_'+self.dataset_name+'_'+self.explainer_name+".png")

        plt.rcParams.update({'font.size': 15})
        # sns.set(font_scale=1.25)
        # sns.set_context("talk")
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})

        g = cor_matrix(df, self.dataset_name+'_'+self.explainer_name)

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        rdd = np.mean(rdd_scores)
        rdd_std = np.std(rdd_scores)
        nll = np.mean(nll_scores)
        nll_std = np.std(nll_scores)
        return auc, auc_std, rdd, rdd_std, nll, nll_std

    def show_results(self):
        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/logs/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())
        auc_scores = []
        re_auc_scores = []
        # binary_auc_scores = []
        nll_scores = []
        re_nll_scores = []
        biers_scores = []
        re_biers_scores = []
        ece_scores = []
        re_ece_scores = []
        for s in self.seeds:
            file_name = str(s) + '_results.json'
            data = load_confidence_results(folder_path, file_name)
            # auc_score, rdd_score, nll_score = data[1], data[2], data[3]
            explanations = data[4]
            ground_truth = self.dataset_loader.edge_ground_truth

            y_true, y_pred = self.prepare_evaluate(explanations)
            # print('y_true: ', y_true)
            # print('y_pred: ', y_pred)
            preds = np.array(y_pred, dtype=np.float32).tolist()
            gts = np.array(y_true, dtype=np.float32).tolist()
            # print('gts: ', gts)
            # print('preds: ', preds)
            auc_score = self.evaluate_auc(gts, preds)
            auc_scores.append(auc_score)
            nll_score = self.evaluate_nll(gts, preds)
            nll_scores.append(nll_score)
            biers_score = self.evaluate_brier_score(gts, preds)
            biers_scores.append(biers_score)
            ece_score, mce = self.evaluate_ece_mce_score(gts, preds)
            ece_scores.append(ece_score)

            preds = [1.0 - i for i in preds]
            re_auc = self.evaluate_auc(gts, preds)
            # re_auc_scores.append(re_auc)
            re_nll = self.evaluate_nll(gts, preds)
            # re_nll_scores.append(re_nll)
            re_bi = self.evaluate_brier_score(gts, preds)
            # re_biers_scores.append(re_bi)
            re_ece, mce = self.evaluate_ece_mce_score(gts, preds)
            # re_ece_scores.append(re_ece)

            if re_auc < auc_score:
                re_auc_scores.append(auc_score)
                re_nll_scores.append(nll_score)
                re_biers_scores.append(biers_score)
                re_ece_scores.append(ece_score)
            else:
                re_auc_scores.append(re_auc)
                re_nll_scores.append(re_nll)
                re_biers_scores.append(re_bi)
                re_ece_scores.append(re_ece)

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        nll = np.mean(nll_scores)
        nll_std = np.std(nll_scores)
        biers = np.mean(biers_scores)
        biers_std = np.std(biers_scores)
        ece = np.mean(ece_scores)
        ece_std = np.std(ece_scores)

        re_auc = np.mean(re_auc_scores)
        re_auc_std = np.std(re_auc_scores)
        re_nll = np.mean(re_nll_scores)
        re_nll_std = np.std(re_nll_scores)
        re_biers = np.mean(re_biers_scores)
        re_biers_std = np.std(re_biers_scores)
        re_ece = np.mean(re_ece_scores)
        re_ece_std = np.std(re_ece_scores)

        print("Auc score: %.4f (%.4f)" % (auc, auc_std))
        print("NLL score: %.4f (%.4f)" % (nll, nll_std))
        print("BIERS score: %.4f (%.4f)" % (biers, biers_std))
        print("ECE score: %.4f (%.4f)" % (ece, ece_std))

        print("Reversed Auc score: %.4f (%.4f)" % (re_auc, re_auc_std))
        print("Reversed NLL score: %.4f (%.4f)" % (re_nll, re_nll_std))
        print("Reversed BIERS score: %.4f (%.4f)" % (re_biers, re_biers_std))
        print("Reversed ECE score: %.4f (%.4f)" % (re_ece, re_ece_std))

        try:
            self.evaluating()
        except:
            print('confidence interval not available...\n')

        self.calc_coefficient()

        return {"auc": auc, "auc_std": auc_std,
                're_auc': re_auc, "re_auc_std": re_auc_std}

        print('corr coef...')
        coef_auc_nll = np.corrcoef(auc_scores, nll_scores)[0, 1]
        coef_auc_biers = np.corrcoef(auc_scores, biers_scores)[0, 1]
        coef_auc_ece = np.corrcoef(auc_scores, ece_scores)[0, 1]

        coef_re_auc_nll = np.corrcoef(re_auc_scores, re_nll_scores)[0, 1]
        coef_re_auc_biers = np.corrcoef(re_auc_scores, re_biers_scores)[0, 1]
        coef_re_auc_ece = np.corrcoef(re_auc_scores, re_ece_scores)[0, 1]

        print("Correlation coefficient between AUC and NLL: %.4f" % coef_auc_nll)
        print("Correlation coefficient between AUC and BIERS: %.4f" % coef_auc_biers)
        print("Correlation coefficient between AUC and ECE: %.4f" % coef_auc_ece)

        print("Correlation coefficient between Reversed AUC and NLL: %.4f" % coef_re_auc_nll)
        print("Correlation coefficient between Reversed AUC and BIERS: %.4f" % coef_re_auc_biers)
        print("Correlation coefficient between Reversed AUC and ECE: %.4f" % coef_re_auc_ece)

        print('spearman coef...')
        correlation_auc_nll, _ = spearmanr(auc_scores, nll_scores)
        correlation_auc_biers, _ = spearmanr(auc_scores, biers_scores)
        correlation_auc_ece, _ = spearmanr(auc_scores, ece_scores)

        correlation_re_auc_nll, _ = spearmanr(re_auc_scores, re_nll_scores)
        correlation_re_auc_biers, _ = spearmanr(re_auc_scores, re_biers_scores)
        correlation_re_auc_ece, _ = spearmanr(re_auc_scores, re_ece_scores)

        print("Spearman correlation between auc nll: ", correlation_auc_nll)
        print("Spearman correlation between auc biers: ", correlation_auc_biers)
        print("Spearman correlation between auc ece: ", correlation_auc_ece)

        print("Spearman correlation between re auc nll: ", correlation_re_auc_nll)
        print("Spearman correlation between re auc biers: ", correlation_re_auc_biers)
        print("Spearman correlation between re auc ece: ", correlation_re_auc_ece)

        return {"auc": auc, "auc_std": auc_std, "nll": nll, "nll_std": nll_std,
                "biers": biers, "biers_std": biers_std, "ece": ece, "ece_std": ece_std}

    def calc_coefficient(self):
        print('Calculating correlation coefficient in graph level...')
        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/logs/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())
        auc_scores = []
        re_auc_scores = []
        # binary_auc_scores = []
        nll_scores = []
        re_nll_scores = []
        biers_scores = []
        re_biers_scores = []
        ece_scores = []
        re_ece_scores = []
        for s in self.seeds:
            s_auc_scores = []
            s_re_auc_scores = []
            s_nll_scores = []
            s_re_nll_scores = []
            s_biers_scores = []
            s_re_biers_scores = []
            s_ece_scores = []
            s_re_ece_scores = []

            file_name = str(s) + '_results.json'
            data = load_confidence_results(folder_path, file_name)
            # auc_score, rdd_score, nll_score = data[1], data[2], data[3]
            explanations = data[4]
            ground_truth = self.dataset_loader.edge_ground_truth

            for exp in explanations:
                y_true, y_pred = self.prepare_evaluate([exp])
                # print('y_true: ', y_true)
                # print('y_pred: ', y_pred)
                preds = np.array(y_pred, dtype=np.float32).tolist()
                gts = np.array(y_true, dtype=np.float32).tolist()
                # print('gts: ', gts)
                # print('preds: ', preds)
                try:
                    nll_score = self.evaluate_nll(gts, preds)
                    nll_scores.append(nll_score)
                    auc_score = self.evaluate_auc(gts, preds)
                    auc_scores.append(auc_score)
                    biers_score = self.evaluate_brier_score(gts, preds)
                    biers_scores.append(biers_score)
                    ece_score, mce = self.evaluate_ece_mce_score(gts, preds)
                    ece_scores.append(ece_score)

                    preds = [1.0 - i for i in preds]
                    re_auc = self.evaluate_auc(gts, preds)
                    # re_auc_scores.append(re_auc)
                    re_nll = self.evaluate_nll(gts, preds)
                    # re_nll_scores.append(re_nll)
                    re_bi = self.evaluate_brier_score(gts, preds)
                    # re_biers_scores.append(re_bi)
                    re_ece, mce = self.evaluate_ece_mce_score(gts, preds)
                    # re_ece_scores.append(re_ece)

                    if re_auc < auc_score:
                        re_auc_scores.append(auc_score)
                        re_nll_scores.append(nll_score)
                        re_biers_scores.append(biers_score)
                        re_ece_scores.append(ece_score)
                    else:
                        re_auc_scores.append(re_auc)
                        re_nll_scores.append(re_nll)
                        re_biers_scores.append(re_bi)
                        re_ece_scores.append(re_ece)
                    # s_auc_scores.append(np.mean(auc_score))
                    # s_nll_scores.append(np.mean(nll_score))
                    # s_biers_scores.append(np.mean(biers_score))
                    # s_ece_scores.append(np.mean(ece_score))
                    #
                    # if re_auc < auc_score:
                    #     s_re_auc_scores.append(np.mean(auc_score))
                    #     s_re_nll_scores.append(np.mean(nll_score))
                    #     s_re_biers_scores.append(np.mean(biers_score))
                    #     s_re_ece_scores.append(np.mean(ece_score))
                    # else:
                    #     s_re_auc_scores.append(np.mean(re_auc))
                    #     s_re_nll_scores.append(np.mean(re_nll))
                    #     s_re_biers_scores.append(np.mean(re_bi))
                    #     s_re_ece_scores.append(np.mean(re_ece))
                except:
                    continue

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        nll = np.mean(nll_scores)
        nll_std = np.std(nll_scores)
        biers = np.mean(biers_scores)
        biers_std = np.std(biers_scores)
        ece = np.mean(ece_scores)
        ece_std = np.std(ece_scores)

        re_auc = np.mean(re_auc_scores)
        re_auc_std = np.std(re_auc_scores)
        re_nll = np.mean(re_nll_scores)
        re_nll_std = np.std(re_nll_scores)
        re_biers = np.mean(re_biers_scores)
        re_biers_std = np.std(re_biers_scores)
        re_ece = np.mean(re_ece_scores)
        re_ece_std = np.std(re_ece_scores)

        print("Auc score: %.4f (%.4f)" % (auc, auc_std))
        print("NLL score: %.4f (%.4f)" % (nll, nll_std))
        print("BIERS score: %.4f (%.4f)" % (biers, biers_std))
        print("ECE score: %.4f (%.4f)" % (ece, ece_std))

        print("Reversed Auc score: %.4f (%.4f)" % (re_auc, re_auc_std))
        print("Reversed NLL score: %.4f (%.4f)" % (re_nll, re_nll_std))
        print("Reversed BIERS score: %.4f (%.4f)" % (re_biers, re_biers_std))
        print("Reversed ECE score: %.4f (%.4f)" % (re_ece, re_ece_std))

        print('corr coef...')
        coef_auc_nll = np.corrcoef(auc_scores, nll_scores)[0, 1]
        coef_auc_biers = np.corrcoef(auc_scores, biers_scores)[0, 1]
        coef_auc_ece = np.corrcoef(auc_scores, ece_scores)[0, 1]

        coef_re_auc_nll = np.corrcoef(re_auc_scores, re_nll_scores)[0, 1]
        coef_re_auc_biers = np.corrcoef(re_auc_scores, re_biers_scores)[0, 1]
        coef_re_auc_ece = np.corrcoef(re_auc_scores, re_ece_scores)[0, 1]

        print("Correlation coefficient between AUC and NLL: %.4f" % coef_auc_nll)
        print("Correlation coefficient between AUC and BIERS: %.4f" % coef_auc_biers)
        print("Correlation coefficient between AUC and ECE: %.4f" % coef_auc_ece)

        print("Correlation coefficient between Reversed AUC and NLL: %.4f" % coef_re_auc_nll)
        print("Correlation coefficient between Reversed AUC and BIERS: %.4f" % coef_re_auc_biers)
        print("Correlation coefficient between Reversed AUC and ECE: %.4f" % coef_re_auc_ece)

        print('spearman coef...')
        correlation_auc_nll, _ = spearmanr(auc_scores, nll_scores)
        correlation_auc_biers, _ = spearmanr(auc_scores, biers_scores)
        correlation_auc_ece, _ = spearmanr(auc_scores, ece_scores)

        correlation_re_auc_nll, _ = spearmanr(re_auc_scores, re_nll_scores)
        correlation_re_auc_biers, _ = spearmanr(re_auc_scores, re_biers_scores)
        correlation_re_auc_ece, _ = spearmanr(re_auc_scores, re_ece_scores)

        print("Spearman correlation between auc nll: ", correlation_auc_nll)
        print("Spearman correlation between auc biers: ", correlation_auc_biers)
        print("Spearman correlation between auc ece: ", correlation_auc_ece)

        print("Spearman correlation between re auc nll: ", correlation_re_auc_nll)
        print("Spearman correlation between re auc biers: ", correlation_re_auc_biers)
        print("Spearman correlation between re auc ece: ", correlation_re_auc_ece)

        try:
            self.evaluating()
        except:
            print('confidence interval not available...\n')

        return {"auc": auc, "auc_std": auc_std, "nll": nll, "nll_std": nll_std,
                "biers": biers, "biers_std": biers_std, "ece": ece, "ece_std": ece_std}


def main():
    # confident_eva = ConfidenceEvaluator()
    m = torch.nn.LogSoftmax(dim=1)
    loss = torch.nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    print('0:')
    print(output)

    print('1:')
    input = torch.tensor([[1., 0., 0., 0., 0.],
                          [1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1.]])
    output = loss(input, target)
    print(output)
    print(m(input))
    output = loss(m(input), target)
    print(output)

    print('2:')
    input = torch.tensor([[0., 1., 0., 0., 0.],
                          [1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1.]])
    output = loss(input, target)
    print(output)
    print(m(input))
    output = loss(m(input), target)
    print(output)

    print('3:')
    input = torch.tensor([[0., 0.9, 0.1, 0., 0.],
                          [1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1.]])
    output = loss(input, target)
    print(output)
    print(m(input))
    output = loss(m(input), target)
    print(output)

    print('4:')
    input = torch.tensor([[0., 0., 1., 0., 0.],
                          [0., 0., 0., 1., 0.],
                          [0.9, 0., 0., 0., 0.1]])
    output = loss(input, target)
    print(output)
    print(m(input))
    output = loss(m(input), target)
    print(output)

    m = torch.nn.Sigmoid()
    loss = torch.nn.BCELoss()
    input = torch.randn(10, requires_grad=True)
    target = torch.empty(10).random_(2)
    output = loss(m(input), target)
    output.backward()
    pass


if __name__ == '__main__':
    main()
