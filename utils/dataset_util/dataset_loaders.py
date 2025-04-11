import json
import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState
from torch_geometric.data import Data
import torch
from utils.dataset_util.data_utils import preprocess_features, preprocess_adj, adj_to_edge_index, load_real_dataset, edge_index_to_adj
from utils.plot_graph import plot_graph, plot_expl
import torch_geometric as ptgeom
from utils.dataset_util.data_utils import get_graph_data
from utils.dataset_util.benzene import Benzene
from utils.dataset_util.fluoride_carbonyl import FluorideCarbonyl
from utils.dataset_util.alkane_carbonyl import AlkaneCarbonyl


class DatasetLoader:
    def __init__(self, dataset_name, input_type, task_type):
        self.data_root_path = '/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/'
        if not os.path.exists(self.data_root_path):
            print('path not working: ', self.data_root_path)
            self.data_root_path = os.getcwd()
            self.data_root_path = os.path.join(self.data_root_path, 'data')
            print('change to: ', self.data_root_path)
        self.dataset_path = os.path.join(self.data_root_path, 'dataset')

        self.dataset_name = dataset_name
        self.input_type = input_type
        self.type = task_type  # reg or cls
        # self.task = 'graph'  # node/graph, if task == graph, parse node dataset large graph into graphs
        self.indices = []
        self.node_indices = []  # mark the central node of the k-hop graph
        self.graphs = []
        self.features = []
        self.labels = []
        self.node_ground_truth = []
        self.edge_ground_truth = []
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []
        pass

    def load_dataset(self, skip_preproccessing=False, shuffle=True):
        """High level function which loads the dataset
        by calling others spesifying in nodes or graphs.

        Keyword arguments:
        :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
        :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
        :param shuffle: Should the returned dataset be shuffled or not.
        :returns: multiple np.arrays
        """
        print(f"Loading {self.dataset_name} dataset")
        if self.dataset_name[:3] == "syn":  # Load node_dataset
            # adj, features, labels, train_mask, val_mask, test_mask = self.load_node_dataset()
            graph, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth = self.load_node_dataset(shuffle)
            # return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
        else:  # Load graph dataset
            graph, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth = self.load_graph_dataset(shuffle)

        self.node_ground_truth = node_ground_truth
        self.edge_ground_truth = edge_ground_truth

        self.indices = [i for i in range(len(graph))]
        if self.task == 'node':
            self.indices = [i for i in range(len(train_mask))]
        self.graphs = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def load_node_dataset(self, shuffle):
        """Load a node dataset.

        :param shuffle:
        :param _dataset: Which dataset to load. Choose from "syn1: ba-shapes", "syn2: ba-com", "syn3: ba-tree" or "syn4: ba-circle"
        :returns: np.array
        """
        path = os.path.join(self.dataset_path, self.dataset_name + '.pkl')

        with open(path, 'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)
        labels = y_train
        labels[val_mask] = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]

        n_graphs = len(train_mask)
        graph = preprocess_adj(adj)[0].astype('int64').T
        graph = torch.LongTensor(graph)
        feature = preprocess_features(features).astype('float32')
        features = []
        if self.task == 'graph':
            node_ground_truth = []
            edge_ground_truth = []
            edge_index = []
            for n in range(n_graphs):
                features.append(feature)
                k_hop_graph = ptgeom.utils.k_hop_subgraph(n, 3, graph)[1].numpy()
                edge_index.append(k_hop_graph)
                edge_label = []
                node_label = []
                for pair in k_hop_graph.T:
                    r = pair[0]
                    c = pair[1]
                    # print(r, c)
                    ele = adj[r][c]
                    if ele == 1:
                        edge_label.append(1)
                        node_label.append(r)
                        node_label.append(c)
                    else:
                        edge_label.append(0)
                node_ground_truth.append(list(set(node_label)))
                edge_ground_truth.append(edge_label)
                pass
        elif self.task == 'node':
            edge_index = graph
            edge_labels = []
            node_ground_truth = []
            for pair in graph.T:
                edge_labels.append(edge_label_matrix[pair[0], pair[1]])
                if edge_label_matrix[pair[0], pair[1]] == 1:
                    node_ground_truth.append(pair[0])
                    node_ground_truth.append(pair[1])
            edge_ground_truth = np.array(edge_labels)
            node_ground_truth = list(set(node_ground_truth))
            features = feature
        else:
            assert 0
        if self.type == 'cls':
            if self.task != 'node':
                labels = np.argmax(labels, axis=1)
        elif self.type == 'reg':
            pass
        else:
            assert 0

        return edge_index, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth
        # return adj, features, labels, train_mask, val_mask, test_mask

    def load_graph_dataset(self, shuffle=True):
        """Load a graph dataset and optionally shuffle it.
        :param shuffle: Boolean. Whether to shuffle the loaded dataset.
        :returns: np.array
        """

        # Load the chosen dataset from the pickle file.
        if self.dataset_name == "mutag":
            path = self.dataset_path + "Mutagenicity" + '.pkl'
            if not os.path.exists(path):  # pkl not yet created
                print("Mutag dataset pickle is not yet created, doing this now. Can take some time")
                adjs, features, labels = load_real_dataset(path, self.dataset_path + '/Mutagenicity/Mutagenicity_')
                print("Done with creating the mutag dataset")
            else:
                adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_mutag_ground_truth(path)
        elif self.dataset_name in ['ba2motif']:
            path = os.path.join(self.dataset_path, self.dataset_name + '.pkl')
            adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_ba2_ground_truth(path)
        elif self.dataset_name in ['benzene']:
            adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_benzene()
        elif self.dataset_name in ['flca']:
            adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_flca()
        elif self.dataset_name in ['alca']:
            adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_alca()
        else:
            print("Unknown dataset")
            raise NotImplemented

        try:
            n_graphs = adjs.shape[0]
        except:
            n_graphs = len(adjs)
        indices = np.arange(0, n_graphs)
        if shuffle:
            prng = RandomState(
                42)  # Make sure that the permutation is always the same, even if we set the seed different
            indices = prng.permutation(indices)

        # Create shuffled data
        try:
            adjs = adjs[indices]
        except:
            adjs = [adjs[i] for i in indices]
        # features = features[indices].astype('float32')
        try:
            features = [features[i].astype('float32') for i in indices]
        except:
            features = [features[i] for i in indices]

        try:
            labels = labels[indices]
        except:
            labels = [labels[i] for i in indices]

        node_ground_truth = [node_ground_truth[i] for i in indices]
        edge_ground_truth = [edge_ground_truth[i] for i in indices]

        # Create masks
        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)
        train_mask = np.full(n_graphs, False, dtype=bool)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False, dtype=bool)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False, dtype=bool)
        test_mask[test_indices] = True

        if self.dataset_name == 'ba2-2motifs':
            self.input_type = 'dense'

        if self.input_type == 'sparse':
            edge_index = adj_to_edge_index(adjs)
        elif self.input_type == 'dense':
            edge_index = adjs
        else:
            assert 0

        # float 64 to float 32
        edge_index = [torch.LongTensor(edge_index[i]) for i in range(len(edge_index))]
        return edge_index, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth

    def load_bareg_dataset_ground_truth(self, adjs, ground_truths):
        """Load the ground truth from the bareg dataset.
        :returns: np.array
        """

        n_graphs = adjs.shape[0]
        indices = np.arange(0, n_graphs)

        # Create shuffled data
        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []

        # Obtain the edge labels.
        insert = 20
        skip = 5
        for i in range(len(edge_indices)):
            edge_index = edge_indices[i]
            ground_truth = ground_truths[i]
            ground_truth = [j for j in ground_truth if j != 0]
            np_node_labels.append(np.array(ground_truth))
            # print(edge_index, ground_truth)
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # print(r, c)
                if r in ground_truth and c in ground_truth:
                    labels.append(1)
                else:
                    labels.append(0)

            # print(labels)
            # assert 0
            np_edge_labels.append(np.array(labels))

        return np.array(np_edge_labels), np.array(np_node_labels)

    def load_ba2_2motifs_dataset_with_ground_truth(self, path):
        """Load the ground truth from the bareg dataset.
        :returns: np.array
        """
        with open(path, 'rb') as fin:
             data = pkl.load(fin)
        adjs = []
        features = []
        labels = []
        node_ground_truth = []
        edge_ground_truth = []
        for i in data:
            adjs.append(np.array(i[0]).T)
            features.append(np.array(i[1]))
            labels.append([1-i[2], i[2]])
            node_ground_truth.append(i[3])
            edge_ground_truth.append(i[4])

        return adjs, features, np.array(labels), node_ground_truth, edge_ground_truth

    def load_bareg3_dataset_ground_truth(self, adjs, ground_truths):
        """Load the ground truth from the bareg dataset.
        :returns: np.array
        """
        n_graphs = len(adjs)
        indices = np.arange(0, n_graphs)

        # Create shuffled data
        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []

        for i in range(len(edge_indices)):
            edge_index = edge_indices[i]
            ground_truth = ground_truths[i]
            ground_truth = [j for j in ground_truth if j != 0]
            np_node_labels.append(np.array(ground_truth))
            # print(edge_index, ground_truth)
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # print(r, c)
                if r in ground_truth and c in ground_truth:
                    labels.append(1)
                else:
                    labels.append(0)

            # print(labels)
            # assert 0
            np_edge_labels.append(np.array(labels))

        return np.array(np_edge_labels), np.array(np_node_labels)

    def load_mutag_ground_truth(self, path):
        new_path = os.path.join(self.dataset_path, 'mutag_with_ground_truth.pkl')
        if os.path.exists(new_path):
            with open(new_path, 'rb') as fin:
                adjs, features, labels, edge_ground_truth, node_ground_truth = pkl.load(fin)
            return adjs, features, labels, edge_ground_truth, node_ground_truth

        with open(path, 'rb') as fin:
            adjs, features, labels = pkl.load(fin)

        print("Loading MUTAG groundtruth, this can take a while")
        path = self.dataset_path + '/Mutagenicity/Mutagenicity_'
        edge_lists, _, edge_label_lists, _ = get_graph_data(path)
        edge_index = adj_to_edge_index(adjs)

        edge_ground_truth = []
        node_ground_truth = []
        for i in range(len(edge_lists)):
            edge_label = []

            for j in range(len(edge_index[i][0])):
                mark = 0
                pair = (edge_index[i][0][j], edge_index[i][1][j])
                for k in range(len(edge_lists[i])):
                    pair2 = edge_lists[i][k]
                    if pair == pair2:
                        edge_label.append(edge_label_lists[i][k])
                        mark = 1
                        break
                if not mark:
                    edge_label.append(0)
            edge_ground_truth.append(edge_label)

            node_label = []
            for j in range(len(edge_label_lists[i])):
                if edge_label_lists[i][j] == 1:
                    node_label.append(edge_lists[i][j][0])
                    node_label.append(edge_lists[i][j][1])
            node_ground_truth.append(list(set(node_label)))

        with open(new_path, 'wb') as fout:
            pkl.dump((adjs, features, labels, edge_ground_truth, node_ground_truth), fout)
        # return shuffled_edge_index, shuffled_labels, shuffled_edge_list, shuffled_edge_label_lists
        return adjs, features, labels, edge_ground_truth, node_ground_truth

    def load_ba2_ground_truth(self, path):
        """Load a the ground truth from the ba2motif dataset.

        :param path: path to data
        :returns: np.array, np.array
        """
        with open(path, 'rb') as fin:
            adjs, features, graph_labels = pkl.load(fin)

        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []
        # Obtain the edge labels.
        insert = 20
        skip = 5
        for edge_index in edge_indices:
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # In line with the original PGExplainer code we determine the ground truth based on the location in the index
                if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                    labels.append(1)
                else:
                    labels.append(0)
            np_edge_labels.append(np.array(labels))
            np_node_labels.append(np.array([20, 21, 22, 23, 24]))
        return adjs, features, graph_labels, np.array(np_edge_labels), np.array(np_node_labels)

    def load_benzene(self):
        file_path = os.path.join(self.dataset_path, 'dataverse_files', 'benzene.npz')
        benzene = Benzene(data_path=file_path)
        explanations = benzene.explanations
        adjs = []
        features = []
        labels = []
        node_ground_truth = []
        edge_ground_truth = []
        for i in range(len(explanations)):
            edge_index = benzene.graphs[i].edge_index.detach().numpy()
            adj = edge_index_to_adj(edge_index, benzene.graphs[i].num_nodes)
            edge_attr = benzene.graphs[i].edge_attr.detach().numpy()
            node_feature = benzene.graphs[i].x.detach().numpy()
            label = benzene.graphs[i].y.detach().numpy().tolist()[0]
            label = [1-label, label]

            edge_gt = np.zeros((edge_index.shape[1]))
            node_gt = np.zeros((node_feature.shape[0]))
            for expl in explanations[i]:
                tmp = expl.edge_imp.detach().numpy()
                edge_gt = edge_gt + tmp
                tmp = expl.node_imp.detach().numpy()
                node_gt = node_gt + tmp
            adjs.append(adj)
            features.append(node_feature)
            labels.append(label)
            node_gt = [1 if node_gt[i] > 0 else 0 for i in range(len(node_gt))]
            edge_gt = [1 if edge_gt[i] > 0 else 0 for i in range(len(edge_gt))]
            node_ground_truth.append(node_gt)
            edge_ground_truth.append(edge_gt)
        # print('labels: ', len(labels), np.sum(np.array(labels)))
        return np.array(adjs), features, np.array(labels), edge_ground_truth, node_ground_truth

    def load_flca(self):
        file_path = os.path.join(self.dataset_path, 'dataverse_files', 'fluoride_carbonyl.npz')
        flca = FluorideCarbonyl(data_path=file_path)
        explanations = flca.explanations
        adjs = []
        features = []
        labels = []
        node_ground_truth = []
        edge_ground_truth = []
        for i in range(len(explanations)):
            edge_index = flca.graphs[i].edge_index.detach().numpy()
            adj = edge_index_to_adj(edge_index, flca.graphs[i].num_nodes)
            edge_attr = flca.graphs[i].edge_attr.detach().numpy()
            node_feature = flca.graphs[i].x.detach().numpy()
            label = flca.graphs[i].y.detach().numpy().tolist()[0]
            label = [1-label, label]

            edge_gt = np.zeros((edge_index.shape[1]))
            node_gt = np.zeros((node_feature.shape[0]))
            for expl in explanations[i]:
                tmp = expl.edge_imp.detach().numpy()
                edge_gt = edge_gt + tmp
                tmp = expl.node_imp.detach().numpy()
                node_gt = node_gt + tmp
            adjs.append(adj)
            features.append(node_feature)
            labels.append(label)
            node_gt = [1 if node_gt[i] > 0 else 0 for i in range(len(node_gt))]
            edge_gt = [1 if edge_gt[i] > 0 else 0 for i in range(len(edge_gt))]
            node_ground_truth.append(node_gt)
            edge_ground_truth.append(edge_gt)
        # print('labels: ', len(labels), np.sum(np.array(labels)))
        return np.array(adjs), features, np.array(labels), edge_ground_truth, node_ground_truth

    def load_alca(self):
        file_path = os.path.join(self.dataset_path, 'dataverse_files', 'alkane_carbonyl.npz')
        alca = AlkaneCarbonyl(data_path=file_path)
        explanations = alca.explanations
        adjs = []
        features = []
        labels = []
        node_ground_truth = []
        edge_ground_truth = []
        for i in range(len(explanations)):
            edge_index = alca.graphs[i].edge_index.detach().numpy()
            adj = edge_index_to_adj(edge_index, alca.graphs[i].num_nodes)
            edge_attr = alca.graphs[i].edge_attr.detach().numpy()
            node_feature = alca.graphs[i].x.detach().numpy()
            label = alca.graphs[i].y.detach().numpy().tolist()[0]
            label = [1-label, label]

            edge_gt = np.zeros((edge_index.shape[1]))
            node_gt = np.zeros((node_feature.shape[0]))
            for expl in explanations[i]:
                tmp = expl.edge_imp.detach().numpy()
                edge_gt = edge_gt + tmp
                tmp = expl.node_imp.detach().numpy()
                node_gt = node_gt + tmp
            adjs.append(adj)
            features.append(node_feature)
            labels.append(label)
            node_gt = [1 if node_gt[i] > 0 else 0 for i in range(len(node_gt))]
            edge_gt = [1 if edge_gt[i] > 0 else 0 for i in range(len(edge_gt))]
            node_ground_truth.append(node_gt)
            edge_ground_truth.append(edge_gt)
        # print('labels: ', len(labels), np.sum(np.array(labels)))
        return np.array(adjs), features, np.array(labels), edge_ground_truth, node_ground_truth

    def create_data_list(self):
        """
        Convert the numpy data to torch tensors and save them in a list.
        :params mask: mask, used to filter the data
        :retuns: list; contains the dataset
        """
        if self.task == 'node':
            x = torch.tensor(self.features)
            edge_index = torch.tensor(self.graphs)

            if self.type == 'cls':
                y = torch.tensor(self.labels, dtype=torch.long)
            else:
                y = torch.tensor(self.labels)
            edge_mask = torch.ones(edge_index.size(1))
            data = Data(x=x, edge_index=edge_index, edge_mask=edge_mask, y=y)
            self.train_data_list = data
            return

        def create_with_mask(mask):
            indices = np.argwhere(mask).squeeze()
            data_list = []
            for i in indices:
                x = torch.tensor(self.features[i])
                edge_index = torch.tensor(self.graphs[i])

                if self.type == 'cls':
                    y = torch.tensor(self.labels[i].argmax(), dtype=torch.long)
                else:
                    y = torch.tensor(self.labels[i])
                edge_mask = torch.ones(edge_index.size(1))
                data = Data(x=x, edge_index=edge_index, edge_mask=edge_mask, y=y)
                data_list.append(data)
                # print(data)
            return data_list

        self.train_data_list = create_with_mask(self.train_mask)
        self.val_data_list = create_with_mask(self.val_mask)
        self.test_data_list = create_with_mask(self.test_mask)

    def plot(self):
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)
        else:
            graph = self.graphs
        label = self.labels
        plot_graph(graph, label)

    def plot_expl(self, idx, expl, save_path=None, file_name=None, if_save=False, if_show=True):
        if if_save:
            if_show = False
        feature = self.features[idx]
        label = self.labels[idx]
        edge_ground_truth = self.edge_ground_truth[idx]
        node_ground_truth = self.node_ground_truth[idx]
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)[idx]
        else:
            graph = self.graphs[idx]
        plot_expl(graph, feature, label, edge_ground_truth, expl, save_path=save_path, file_name=file_name, if_save=if_save, if_show=if_show)
        # assert 0

    def save_expl(self, idx, expl, auc, save_path=None, file_name=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = idx.tolist()
        expl = expl.tolist()
        try:
            auc = auc.tolist()
        except:
            auc = float(auc)
        try:
            feature = self.features[idx].tolist()
        except:
            feature = self.features[idx]
        try:
            label = self.labels[idx].tolist()
        except:
            label = self.labels[idx]
        edge_ground_truth = self.edge_ground_truth[idx].tolist()
        node_ground_truth = self.node_ground_truth[idx].tolist()
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)[idx].tolist()
        else:
            graph = self.graphs[idx].tolist()

        result = {
            'idx': idx,
            'graph': graph,
            'feature': feature,
            'label': label,
            'edge_ground_truth': edge_ground_truth,
            'node_ground_truth': node_ground_truth,
            'expl': expl,
            'auc': auc,
        }

        with open(os.path.join(save_path, file_name), 'w') as f:
            json.dump(result, f)

    def plot_expl_from_json(self, explainer_name, dataset_name, model_name, seed, idx, if_save=False, if_show=True):
        if if_save:
            if_show = False

        folder_path = os.path.join('data/results/visualization', explainer_name, dataset_name, model_name)

        for config in os.listdir(folder_path):
            config_path = os.path.join(folder_path, config)

            for i in os.listdir(config_path):
                if i.startswith(str(seed)+'_'):
                    config_path = os.path.join(config_path, i)
            for i in os.listdir(config_path):
                if i.startswith(str(idx)+'_'):
                    config_path = os.path.join(config_path, i)

            file_name = str(idx)+'.json'
            file_path = os.path.join(config_path, file_name)

            with open(file_path, 'r') as f:
                data = json.load(f)

            graph = data['graph']
            feature = data['feature']
            label = data['label'][0]
            edge_ground_truth = data['edge_ground_truth']
            node_ground_truth = data['node_ground_truth']
            expl = data['expl']

            plot_expl(graph, feature, label, edge_ground_truth, expl, save_path=config_path, file_name=str(idx)+'.png',
                      if_save=if_save, if_show=if_show)
        # assert 0

    def show_results_(self, explainer_name, dataset_name, model_name):
        # bad method, only for test
        data = self.load_results(explainer_name, dataset_name, model_name)
        return data[1], data[2], data[3], data[4]

    def load_results_(self, explainer_name, dataset_name, model_name):
        # bad method, only for test
        # folder_path = os.path.join('/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/visualization',
        #                            explainer_name, dataset_name, model_name)
        folder_path = os.path.join(self.data_root_path, 'results/visualization', explainer_name, dataset_name, model_name)
        confs = os.listdir(folder_path)
        best_auc = -1
        data = []
        for conf in confs:
            conf_path = os.path.join(folder_path, conf)
            seeds = os.listdir(conf_path)
            auc_s = []
            rdd_s = []
            for seed in seeds:
                if seed == '.DS_Store':
                    continue
                res = seed.split('_')
                idx = res[0]
                auc = float(res[1])
                rdd = float(res[2])
                auc_s.append(auc)
                rdd_s.append(rdd)
            auc = np.mean(auc_s)
            auc_std = np.std(auc_s)
            rdd = np.mean(rdd_s)
            rdd_std = np.std(rdd_s)
            if auc > best_auc:
                best_auc = auc
                data = [conf, auc, auc_std, rdd, rdd_std]
        return data
