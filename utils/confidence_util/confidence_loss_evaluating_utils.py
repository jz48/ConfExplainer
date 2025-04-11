import numpy
import json
import os
from matplotlib import pyplot as plt


class evaluator(object):
    def __init__(self, save_path=None, file_name=None):
        self.save_path = '/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/confidence_evaluation/logs'
        self.explainer_name = 'CONF2PGE'
        self.explainer_name = 'PGE'
        self.dataset_name = 'ba2motif'
        self.model_name = 'GraphGCN'
        self.best_log = None
        self.dir_path = None
        self.logs = None

    def load_data(self, path):
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.loads(f.readline())
        return data

    def plot_data(self, loss_log, file_name, file_path=None):
        if os.path.exists(file_path) and file_path is not None:
            return
        data = loss_log['0']
        x = [i+1 for i in range(len(data['conf_loss']))]
        conf_loss = [i[1] for i in data['conf_loss']]
        explainer_loss = [i[1] for i in data['expl_loss']]
        total_loss = [conf_loss[i] + explainer_loss[i] for i in range(len(conf_loss))]
        auc_scores = [i[1] for i in data['scores']]
        auc_scores = [auc_scores[i] for i in range(len(auc_scores)) if i % 2 == 1]
        acc_scores = [i[2] for i in data['scores']]
        acc_scores = [acc_scores[i] for i in range(len(acc_scores)) if i % 2 == 1]

        # draw loss and score curves
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(x, conf_loss, color='red', label='Confidence Loss')
        ax1.plot(x, explainer_loss, color='blue', label='Explainer Loss')
        ax1.plot(x, total_loss, color='green', label='Total Loss')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Score')
        x = [i+1 for i in range(len(auc_scores))]
        ax2.plot(x, auc_scores, color='orange', label='AUC Score')
        ax2.plot(x, acc_scores, color='purple', label='Binary AUC Score')
        ax2.legend(loc='upper right')
        plt.title(file_name)
        plt.savefig(file_path)
        plt.close()
        # assert 0
        pass

    def process_data(self, loss_log, explainer_name):
        if explainer_name == 'CONF2PGE':
            return loss_log
        elif explainer_name in ['PGE']:
            new_loss_log = {}
            for key in loss_log.keys():
                new_loss_log[key] = {}
                new_loss_log[key]['conf_loss'] = [(i[0], 0.0) for i in loss_log[key]['expl_loss']]
                new_loss_log[key]['expl_loss'] = [(i[0], i[1][0]) for i in loss_log[key]['expl_loss']]
                new_loss_log[key]['scores'] = loss_log[key]['scores']
            return new_loss_log
        else:
            raise NotImplementedError

    def run(self):
        self.dir_path = os.path.join(self.save_path, self.explainer_name, self.dataset_name, self.model_name)
        logs = os.listdir(self.dir_path)

        self.logs = [log for log in logs if ('epochs_100' in log)]
        if self.explainer_name == 'CONF2PGE':
            self.logs = [log for log in logs if ('epochs_100' in log and 'ibcls' not in log)]

        for log in self.logs:
            print(log)
            log_path = os.path.join(self.dir_path, log)
            loss_log_path = os.path.join(log_path, 'loss_log.json')
            loss_log = self.load_data(loss_log_path)
            loss_log = self.process_data(loss_log, self.explainer_name)
            file_name = self.explainer_name + '_' + self.dataset_name + '_' + self.model_name + '_' + log + '.png'
            file_path = os.path.join('/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/figs/loss_log', file_name)
            self.plot_data(loss_log, file_name, file_path)

    def find_best_performance_log(self):
        mark = -1
        for log in self.logs:
            print(log)
            log_path = os.path.join(self.dir_path, log)
            loss_log_path = os.path.join(log_path, 'loss_log.json')
            loss_log = self.load_data(loss_log_path)
            log_auc = loss_log['0']['scores'][-1][1]
            if log_auc > mark:
                mark = log_auc
                self.best_log = log
        print('best log: ', self.best_log, 'auc: ', mark)
        pass

    def compare_PGE_CONFPGE(self):
        explainer_names = ['PGE', 'CONF2PGE']
        file_name = 'compare.png'
        file_path = os.path.join('/Users/path_to_yours/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/figs/loss_log',
                                 file_name)
        # draw four figures to compare PGE and CONFPGE
        fig, axs = plt.subplots(2, 2)
        for explainer_name in explainer_names:
            self.explainer_name = explainer_name
            self.run()
            self.find_best_performance_log()

            loss_log_path = os.path.join(self.dir_path, self.best_log, 'loss_log.json')
            loss_log = self.load_data(loss_log_path)
            loss_log = self.process_data(loss_log, self.explainer_name)

            expl_loss = [i[1] for i in loss_log['0']['expl_loss']]
            x = [i+1 for i in range(len(expl_loss))]
            axs[0][0].plot(x, expl_loss, label=self.explainer_name)

            if self.explainer_name in ['PGE', 'CONFPGE']:
                # skip confidence loss
                total_loss = expl_loss
                pass
            elif self.explainer_name == 'CONF2PGE':
                x = [i+1 for i in range(len(loss_log['0']['conf_loss']))]
                conf_loss = [i[1] for i in loss_log['0']['conf_loss']]
                axs[0][1].plot(x, conf_loss, label=self.explainer_name)
                total_loss = [conf_loss[i] + expl_loss[i] for i in range(len(conf_loss))]

            axs[1][0].plot(x, total_loss, label=self.explainer_name)
            auc_scores = [i[1] for i in loss_log['0']['scores']]
            auc_scores = [auc_scores[i] for i in range(len(auc_scores)) if i % 2 == 1]
            acc_scores = [i[2] for i in loss_log['0']['scores']]
            acc_scores = [acc_scores[i] for i in range(len(acc_scores)) if i % 2 == 1]
            x = [i+1 for i in range(len(auc_scores))]
            axs[1][1].plot(x, auc_scores, label=self.explainer_name)
            axs[1][1].plot(x, acc_scores, label=self.explainer_name)

        axs[0][0].set_title('Explainer Loss')
        axs[0][0].legend(loc='upper right')
        axs[0][1].set_title('Confidence Loss')
        axs[0][1].legend(loc='upper right')
        axs[1][0].set_title('Total Loss')
        axs[1][0].legend(loc='upper right')
        axs[1][1].set_title('AUC Score')
        axs[1][1].legend(loc='upper right')
        plt.savefig(file_path)
        plt.close()
        pass

    def statistic_performance(self):
        self.explainer_name = 'CONF2PGE'
        self.run()
        self.logs = sorted(self.logs)
        for log in self.logs:
            log_path = os.path.join(self.dir_path, log)
            loss_log_path = os.path.join(log_path, 'loss_log.json')
            loss_log = self.load_data(loss_log_path)
            loss_log = self.process_data(loss_log, self.explainer_name)
            auc = 0.0
            acc = 0.0
            for seed in range(10):
                auc_score = loss_log[str(seed)]['scores'][-1][1]
                acc_score = loss_log[str(seed)]['scores'][-1][2]
                auc += auc_score
                acc += acc_score
            auc /= 10
            acc /= 10
            print(log, 'auc: ', numpy.mean(auc), 'acc: ', numpy.mean(acc))


if __name__ == '__main__':
    evaluator = evaluator()
    evaluator.run()
    # evaluator.find_best_performance_log()
    # evaluator.compare_PGE_CONFPGE()
    evaluator.statistic_performance()
