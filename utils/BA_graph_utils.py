import random
import numpy as np
import pickle as pkl
# from experiment_models_training.datasets.dataset_loaders import load_dataset
import networkx as nx
import os

def build_ba2_2motifs(samples=1000, nodes=20, num_edges=1):
    """
    In this dataset, we build static graph dataset ba-2motifs.
    :param samples:
    :param nodes:
    :param num_edges:
    :return:
    """
    data = []
    sum_edges = 0
    sum_graph_edge = 0
    sum_init_edge = 0
    count = 0
    sum_labels = 0

    test_set_a = set()
    test_set_b = set()

    for i in range(samples):
        features = []

        graph = nx.barabasi_albert_graph(n=nodes, m=num_edges, seed=i)
        graph.add_nodes_from([20, 21, 22, 23, 24])
        graph.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])
        graph.add_edge(random.randrange(0, 19), random.randrange(20, 24))
        sum_init_edge += graph.number_of_edges()
        random.seed(i)
        rand_v = random.random()
        if rand_v <= 0.5:
            # circle
            label = 0

            init_node = random.randrange(0, 20)
            while True:
                second_node = random.randrange(0, 20)
                if second_node == init_node:
                    continue
                if graph.has_edge(init_node, second_node):
                    init_node = random.randrange(0, 20)
                    continue
                else:
                    graph.add_edge(init_node, second_node)
                    break

            test_set_a.add(init_node)
            test_set_a.add(second_node)
        else:
            # house
            label = 1

            edge = random.choice([[20, 22], [21, 23], [22, 24], [23, 20], [24, 21]])

            graph.add_edge(edge[0], edge[1])

            test_set_b.add(edge[0])
            test_set_b.add(edge[1])

        sum_graph_edge += graph.number_of_edges()
        for _ in range(25):
            features.append([0.1] * 10)
        ground_truth = [20, 21, 22, 23, 24]
        edge_ground_truth = []

        edges = list(graph.edges)
        # print('single graph edges', edges)
        bi_edges = []
        for edge in edges:
            bi_edges.append([edge[1], edge[0]])
            bi_edges.append([edge[0], edge[1]])
        # sort edges
        bi_edges = sorted(bi_edges, key=lambda x: (x[0], x[1]))
        # print('bi-dr graph edges')
        edges = bi_edges
        # print(edges)

        count += 1
        sum_edges += len(edges)
        sum_labels += label

        for edge in edges:
            # print(edge)
            if edge[0] in ground_truth and edge[1] in ground_truth:
                edge_ground_truth.append(1)
            else:
                edge_ground_truth.append(0)

        data.append([edges, features, label, ground_truth, edge_ground_truth])
    print('test_set_a', test_set_a)
    print('test_set_b', test_set_b)

    data_path = os.path.join('./data/dataset', f"ba2-2motifs.pkl")
    with open(data_path, 'wb') as f:
        pkl.dump(data, f)

    # print(data)
    # average edge numbers
    edge_nums = []
    for i in range(samples):
        edge_nums.append(len(data[i][0][0]))
    print("Average edge numbers: ", sum(edge_nums) / len(edge_nums))
    # average label
    labels = []
    for i in range(samples):
        labels.append(data[i][2])
    print("Average label: ", sum(labels) / len(labels))
    print("Average init graph edge numbers: ", sum_init_edge / samples)
    print("Average graph edge numbers: ", sum_graph_edge / samples)
    print("Average edge numbers: ", sum_edges / count)
    print("Average label: ", sum_labels / count)
    return data


def build_ba_reg(samples=1000, nodes=20, edges=1):
    # rng = random.Random(1234)
    data = []
    data_alt_1 = []  # change features for nodes outside house
    data_alt_2 = []  # remove edges for nodes outside house
    data_alt_3 = []  # larger noisy graph with ground truth to replace part of it
    data_alt_4 = []  # larger noisy graph with ground truth to replace part of it and remove edges and nodes outside
                        # house
    data_alt_5 = []  # change noisy feature to 10x
    data_alt_6 = []  # change noisy feature to 100x
    data_alt_7 = []  # change ground truth feature and label to 10x
    data_alt_8 = []  # change ground truth feature and label to 0.1x
    for i in range(samples):
        graph = nx.barabasi_albert_graph(n=nodes, m=edges, seed=i)
        graph_alt_2 = nx.Graph()
        # print(graph)
        # print(graph.nodes)
        # print(graph.edges)

        graph.add_nodes_from([20, 21, 22, 23, 24])
        graph.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])
        graph_alt_2.add_nodes_from([i for i in range(25)])
        graph_alt_2.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])
        random.seed(i)
        graph.add_edge(random.randrange(0, 19), random.randrange(20, 24))
        features = []
        for _ in range(25):
            features.append([random.randrange(1, 1000) / 10] * 10)
        label = [sum(i[0] * 10 for i in features[20:25])]
        features_alt_1 = features.copy()
        for idx in range(20):
            features_alt_1[idx] = [random.randrange(1, 1000) / 10] * 10
        features_alt_2 = features.copy()
        for idx in range(20):
            features_alt_2[idx] = [0.0] * 10
        ground_truth = [20, 21, 22, 23, 24]

        # alt3: larger noisy graph with ground truth to replace part of it
        graph_alt_3 = nx.barabasi_albert_graph(n=(nodes + 5) * 2, m=edges, seed=i + 1)
        graph_alt_4 = nx.barabasi_albert_graph(n=(nodes + 5) * 2, m=edges, seed=i + 1)
        features_alt_3 = []
        for j in range(25):
            features_alt_3.append(features[j])
        for j in range(25):
            features_alt_3.append([random.randrange(1, 1000) / 10] * 10)
        # print(graph.nodes)
        # print(graph.edges)
        # print(features, label)
        # print(graph_alt_2.nodes)
        # print(graph_alt_2.edges)

        # print(features)
        # print(features_alt_1)
        # print(features_alt_2)
        # assert 0
        data.append([graph, features, label, ground_truth])
        data_alt_1.append([graph, features_alt_1, label, ground_truth])
        data_alt_2.append([graph_alt_2, features_alt_3, label, ground_truth])
        data_alt_3.append([graph_alt_3, features_alt_3, label, ground_truth])
        data_alt_4.append([graph_alt_4, features_alt_3, label, ground_truth])
        # print(data[0][0].edges)
        # print(data_alt_3[0][0].edges)
        edges_3 = []
        edges_4 = []
        for edge in data[i][0].edges:
            # print(edge)
            edges_3.append(edge)
            # print(edges_3)
            if 20 <= edge[0] <= 24 and 20 <= edge[1] <= 24:
                edges_4.append(edge)
        for edge in data_alt_3[i][0].edges:
            if edge[0] > 24 or edge[1] > 24:
                edges_3.append(edge)
                edges_4.append(edge)
        data_alt_3[i][0].edges = edges_3
        data_alt_4[i][0].edges = edges_4
        # print(data_alt_3[0][0].edges)
        # print(data_alt_4[0][0].edges)
        # assert 0

        # modify features and labels
        features_alt_5 = []
        features_alt_6 = []
        features_alt_7 = []
        features_alt_8 = []
        for idx in range(20):
            features_alt_5.append([bit * 10 for bit in features[idx]])
            features_alt_6.append([bit * 100 for bit in features[idx]])
            features_alt_7.append(features[idx])
            features_alt_8.append(features[idx])
        for idx in range(20, 25):
            features_alt_5.append(features[idx])
            features_alt_6.append(features[idx])
            features_alt_7.append([bit * 10 for bit in features[idx]])
            features_alt_8.append([bit * 0.1 for bit in features[idx]])
        label_alt_7 = [labe * 10 for labe in label]
        label_alt_8 = [labe * 0.1 for labe in label]
        data_alt_5.append([graph, features_alt_5, label, ground_truth])
        data_alt_6.append([graph, features_alt_6, label, ground_truth])
        data_alt_7.append([graph, features_alt_7, label_alt_7, ground_truth])
        data_alt_8.append([graph, features_alt_8, label_alt_8, ground_truth])

        for line in range(25):
            pass
            # print(features[line][0], features_alt_5[line][0], features_alt_6[line][0], features_alt_7[line][0], features_alt_8[line][0])
        # print(label)
        # print(label_alt_7)
        # print(label_alt_8)
        # assert 0
    return data, data_alt_1, data_alt_2, data_alt_3, data_alt_4, data_alt_5, data_alt_6, data_alt_7, data_alt_8


def build_ba_reg2(samples=1000, nodes=120, edges=1):
    # rng = random.Random(1234)
    data = []
    for i in range(samples):
        random.seed(i)
        num_to_add = random.randrange(1, 20)
        house0 = nodes - 5 * num_to_add
        graph = nx.barabasi_albert_graph(n=nodes - 5 * num_to_add, m=edges, seed=i)
        for h in range(num_to_add):
            # graph = nx.barabasi_albert_graph(n=120-5*num_to_add, m=edges, seed=i)
            graph.add_nodes_from([house0, house0 + 1, house0 + 2, house0 + 3, house0 + 4])
            graph.add_edges_from([(house0, house0 + 1), (house0 + 1, house0 + 2), (house0 + 2, house0 + 3),
                                  (house0 + 3, house0 + 4), (house0 + 4, house0)])
            graph.add_edge(random.randrange(0, nodes - 5 * num_to_add - 1), random.randrange(house0, house0 + 4))
            house0 += 5
            # if num_to_add < 20:
            #      graph.add_nodes_from([nid for nid in range(house0, house0 + (20 - num_to_add) * 5)])
        features = []
        num_nodes = nodes
        for _ in range(num_nodes):
            features.append([0.1] * 10)
        label = [num_to_add]
        ground_truth = [i for i in range(nodes - 5 * num_to_add, nodes)]
        # print(graph.nodes)
        # print(graph.edges)
        # print(features, label)

        # assert 0
        data.append([graph, features, label, ground_truth])
    return data


def build_ba_reg3(samples=1000, nodes=120, edges=1):
    """ba-reg2 with dynamic number of nodes"""
    # rng = random.Random(1234)
    data = []
    for i in range(samples):
        random.seed(i)
        num_to_add = random.randrange(1, 20)
        num_base = random.randint(20, 80)
        nodes = num_base + 5 * num_to_add
        house0 = num_base
        graph = nx.barabasi_albert_graph(n=num_base, m=edges, seed=i)
        for h in range(num_to_add):
            # graph = nx.barabasi_albert_graph(n=120-5*num_to_add, m=edges, seed=i)
            graph.add_nodes_from([house0, house0 + 1, house0 + 2, house0 + 3, house0 + 4])
            graph.add_edges_from([(house0, house0 + 1), (house0 + 1, house0 + 2), (house0 + 2, house0 + 3),
                                  (house0 + 3, house0 + 4), (house0 + 4, house0)])
            graph.add_edge(random.randrange(0, num_base - 1), random.randrange(house0, house0 + 4))
            house0 += 5
            # if num_to_add < 20:
            #      graph.add_nodes_from([nid for nid in range(house0, house0 + (20 - num_to_add) * 5)])
        features = []
        num_nodes = nodes
        for _ in range(num_nodes):
            features.append([0.1] * 10)
        label = [num_to_add]
        ground_truth = [i for i in range(num_base, num_base + 5 * num_to_add)]
        # print(graph.nodes)
        # print(graph.edges)
        # print(features, label)

        # assert 0
        data.append([graph, features, label, ground_truth])
    return data


def save_ba_reg(data, path, nodes=120):
    graphs = []
    features = []
    labels = []
    ground_truths = []
    for i in data:
        tmp = i[1]
        # if len(i[1]) < 120:
        #     tmp += [[0.0] * 10] * (120 - len(i[1]))
        # print(len(tmp))
        features.append(tmp)
        labels.append(i[2])
        # print(i[0])
        # print(i[0].nodes)
        # print(i[0].edges)
        nodes = len(i[1])
        graph = np.zeros((nodes, nodes), dtype=float)
        for edge in i[0].edges:
            # print(i)
            graph[edge[0], edge[1]] = 1.0
            graph[edge[1], edge[0]] = 1.0
        graphs.append(graph)
        ground_truths.append(i[3] + [0] * (nodes - len(i[3])))
        # a = abcde
        pass
    graphs = graphs
    features = features
    labels = np.asarray(labels, dtype=np.float32)
    ground_truths = ground_truths
    # print(graphs[0], features[0], labels[0])
    with open(path, 'wb') as f:
        pkl.dump((graphs, features, labels, ground_truths), f)
    pass


if __name__ == '__main__':
    # data, data_alt_1, data_alt_2, data_alt_3, data_alt_4, data_alt_5, data_alt_6, data_alt_7, data_alt_8 = build_ba_reg()
    # save_ba_reg(data, './dataset/BA-Reg1.pkl', nodes=25)
    # save_ba_reg(data_alt_1, './dataset/BA-Reg1-alt-1.pkl', nodes=25)
    # save_ba_reg(data_alt_2, './dataset/BA-Reg1-alt-2.pkl', nodes=25)
    # save_ba_reg(data_alt_3, './dataset/BA-Reg1-alt-3.pkl', nodes=50)
    # save_ba_reg(data_alt_4, './dataset/BA-Reg1-alt-4.pkl', nodes=50)
    # save_ba_reg(data_alt_5, './dataset/BA-Reg1-alt-5.pkl', nodes=25)
    # save_ba_reg(data_alt_6, './dataset/BA-Reg1-alt-6.pkl', nodes=25)
    # save_ba_reg(data_alt_7, './dataset/BA-Reg1-alt-7.pkl', nodes=25)
    # save_ba_reg(data_alt_8, './dataset/BA-Reg1-alt-8.pkl', nodes=25)
    # data = build_ba_reg2()
    # save_ba_reg(data, './dataset/BA-Reg2.pkl', nodes=120)
    # data = build_ba_reg3()
    # save_ba_reg(data, '../data/dataset/bareg3.pkl')

    build_ba2_2motifs()
    pass
