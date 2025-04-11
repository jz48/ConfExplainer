import torch
import time
import numpy as np
from utils.confidence_util.confidence_evaluating import ConfidenceEvaluator

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def tune_confpge(config=None):
    # tune the confidence loss weight
    results = {}
    datasets = ['ba2motif', 'benzene', 'flca', 'alca', 'mutag']  #
    conf_loss_weights = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    models = ['GraphGCN']
    explainers = ['CONF5PGE']
    m = models[0]
    e = explainers[0]
    loss_type = 'ibconf5'

    for d in datasets:
        for weight in conf_loss_weights:
            print(f'Running {d} with {m} and {e} with weight {weight}')
            evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                            save_confidence_scores=True)
            evaluator.explainer_manager.explainer.conf_loss_weight = weight
            evaluator.explainer_manager.explainer.epochs = 30
            evaluator.set_experiments()
            # evaluator.evaluating()
            # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
            result = evaluator.show_results()
            results[str([d, weight])] = result

            # save the results
            with open('confpge_tuning_results.txt', 'w') as f:
                for key, value in results.items():
                    f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    tune_confpge()

