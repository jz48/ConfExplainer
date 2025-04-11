import torch
import time
import numpy as np
from utils.confidence_util.confidence_evaluating import ConfidenceEvaluator

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def run_pge_noisy_test():
    print('Doing noisy test...')
    models = ['GraphGCN']
    datasets = ['mutag', 'benzene', 'flca', 'alca', 'ba2motif']  # 'ba2motif', 'mutag', 'benzene, 'flca', 'alca',
    explainers = ['PGE']  # 'PGE', 'CONFPGE', CONF2PGE
    loss_type = 'ib'  # 'ib', 'ce', 'ibconf1', 'ibconf2'
    for d in datasets:
        for e in explainers:
            for m in models:
                print('Dataset: ', d, 'Model: ', m, 'Explainer: ', e)
                evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.epochs = 30
                evaluator.set_experiments()
                evaluator.show_results()
                # evaluator.evaluating()
                for noisy_mod in [4, 5]:  # 1, 2, 3, 4, 5
                    evaluator.evaluate_with_noisy(noisy_mod=noisy_mod)
                    # evaluator.show_results()
                    pass


def run_confpge_noisy_test():
    print('Doing noisy test...')
    models = ['GraphGCN']

    datasets = ['ba2motif']  #'flca' , 'mutag', 'alca', 'ba2motif' 'benzene',
    explainers = ['CONF5PGE', 'CONF6PGE']  # 'PGE', 'CONFPGE', CONF2PGE
    loss_type = 'ibconf5'  # 'ib', 'ce', 'ibconf1', 'ibconf2'
    for e in explainers:
        for d in datasets:
            for m in models:
                evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=True,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.conf_loss_weight = 100
                evaluator.explainer_manager.explainer.epochs = 30
                evaluator.set_experiments()
                # evaluator.evaluating()
                for noisy_mod in [4, 5]:  # 1, 2, 3, 4, 5
                    evaluator.evaluate_with_noisy(noisy_mod=noisy_mod)
                    # evaluator.show_results()
                    pass


def run_confpge():
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene', 'flca', 'alca'] #
    # datasets = ['ba2-2motifs']
    # datasets = ['bareg1', 'bareg2']
    explainers = ['CONF5PGE']  # 'PGE', 'CONFPGE', 'CONF5PGE', 'CONF6PGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ibconf5']:  #'ibconf5', 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2', 'ibconf2_3'
                    for conf_loss_weight in [100]:
                        print('conf_loss_weight: ', conf_loss_weight)
                        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                        save_confidence_scores=True)
                        evaluator.explainer_manager.explainer.conf_loss_weight = conf_loss_weight
                        evaluator.explainer_manager.explainer.epochs = 30
                        evaluator.set_experiments()
                        # evaluator.evaluating()
                        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                        # evaluator.show_results()

def run_time_analyze():
    # This is for time analyze, record time spend for each config
    # and save the time to a file
    print('Doing time analysis...')
    time_recorder = {}
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene', 'flca', 'alca'] #
    # datasets = ['ba2-2motifs']
    # datasets = ['bareg1', 'bareg2']
    explainers = ['CONF5PGE']  # 'PGE', 'CONFPGE', 'CONF5PGE', 'CONF6PGE'

    # concatenate configs into list
    configs = []

    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ibconf5']:  #'ibconf5', 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2', 'ibconf2_3'
                    configs.append((d, m, e, loss_type, 100))


    explainers = ['PGE']
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ib']:  #'ibconf5', 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2', 'ibconf2_3'
                    configs.append((d, m, e, loss_type, 0))

    for config in configs:
        d, m, e, loss_type, conf_loss_weight = config
        print('Dataset: ', d, 'Model: ', m, 'Explainer: ', e, 'Loss type: ', loss_type, 'conf_loss_weight: ', conf_loss_weight)
        time0 = time.time()
        print('conf_loss_weight: ', conf_loss_weight)
        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=True,
                                        save_confidence_scores=True)
        evaluator.explainer_manager.explainer.conf_loss_weight = conf_loss_weight
        evaluator.explainer_manager.explainer.epochs = 30
        evaluator.set_experiments()
        # evaluator.evaluating()
        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
        # evaluator.show_results()
        time1 = time.time()
        time_recorder[(d, m, e, loss_type, conf_loss_weight)] = time1 - time0

        with open('time_recorder.txt', 'w') as f:
            for key, value in time_recorder.items():
                f.write(f'{key}: {value}\n')

    print('Time recorder saved to time_recorder.txt')


def run_ablation():
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene']  # 'ba2motif', 'mutag', 'benzene'

    explainers = ['']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ib']:  # 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2'
                    for conf_loss_weight in [1]:
                        print('conf_loss_weight: ', conf_loss_weight)
                        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                        save_confidence_scores=True)
                        evaluator.explainer_manager.explainer.conf_loss_weight = conf_loss_weight
                        evaluator.explainer_manager.explainer.epochs = 30
                        auc, auc_std, ba, ba_std, nll, nll_std = evaluator.set_experiments()
                        # evaluator.evaluating()
                        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()

                        print(d, m, e,
                              'auc: ', auc, 'auc_std: ', auc_std,
                              'binary auc: ', ba, 'binary_auc_std: ', ba_std,
                              'nll: ', nll, 'nll_std: ', nll_std)

    explainers = ['CONF2PGE']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ibconf2_3_1',
                                  'ibconf2_3']:  # 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2'
                    for conf_loss_weight in [1, 3, 10, 20]:
                        print('conf_loss_weight: ', conf_loss_weight)
                        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                        save_confidence_scores=True)
                        evaluator.explainer_manager.explainer.conf_loss_weight = conf_loss_weight
                        evaluator.explainer_manager.explainer.epochs = 30
                        auc, auc_std, ba, ba_std, nll, nll_std = evaluator.set_experiments()
                        # evaluator.evaluating()
                        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()

                        print(d, m, e,
                              'auc: ', auc, 'auc_std: ', auc_std,
                              'binary auc: ', ba, 'binary_auc_std: ', ba_std,
                              'nll: ', nll, 'nll_std: ', nll_std)


def run_baselines():
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene', 'flca', 'alca']  #
    datasets = ['ba2-2motifs']
    datasets = ['bareg1', 'bareg2']
    explainers = ['GNNE', 'PGE']  # 'PGE', 'CONFPGE', 'CONF5PGE', 'CONF6PGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in [
                    'ib']:  # 'ibconf5', 'ibconf1', 'ibconf1_2', 'ibconf1_3', 'ibconf2', 'ibconf2_2', 'ibconf2_3'
                    for conf_loss_weight in [100]:
                        print('Dataset: ', d, 'Model: ', m, 'Explainer: ', e, 'Loss type: ', loss_type)
                        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                        save_confidence_scores=True)
                        evaluator.explainer_manager.explainer.conf_loss_weight = conf_loss_weight
                        evaluator.explainer_manager.explainer.epochs = 30
                        evaluator.set_experiments()
                        # evaluator.evaluating()
                        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                        # evaluator.show_results()


def run_ensembles():
    # This is for ensemble test, record time spend for each config
    print('Doing ensemble test...')
    models = ['GraphGCN']
    datasets = ['ba2motif', 'alca', 'mutag', 'benzene', 'flca']  # 'alca', 'ba2motif', 'mutag', 'benzene', 'flca', ]  # 'ba2motif', 'mutag'
    # datasets = ['ba2-2motifs']
    # datasets = ['bareg3']
    explainers = ['PGEBE', 'PGEDE']  # 'PGE', 'CONFPGE'
    time_recorder = {}
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ib']:
                    print('Dataset: ', d, 'Model: ', m, 'Explainer: ', e, 'Loss type: ', loss_type)
                    time0 = time.time()
                    evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=True,
                                                    save_confidence_scores=True)
                    evaluator.explainer_manager.explainer.epochs = 30
                    evaluator.set_experiments()
                    time1 = time.time()
                    time_recorder[(d, m, e, loss_type)] = time1 - time0
    with open('time_recorder_ensemble.txt', 'w') as f:
        for key, value in time_recorder.items():
            f.write(f'{key}: {value}\n')



def run_llm():
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene']  # 'ba2motif', 'mutag', 'benzene'

    explainers = []  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                evaluator = ConfidenceEvaluator(d, m, e, loss_type='ib', force_run=False,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.epochs = 100
                evaluator.set_experiments()
                # evaluator.evaluating()
                # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                evaluator.show_results()

    datasets = ['mutag', 'benzene']
    explainers = ['LLMPGE']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                evaluator = ConfidenceEvaluator(d, m, e, loss_type='ib_llm', force_run=True,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.epochs = 100
                evaluator.set_experiments()
                # evaluator.evaluating()
                # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                evaluator.show_results()


def survey_the_extreme_edge_value():
    models = ['GraphGCN']
    datasets = ['ba2motif']  # 'ba2motif', 'mutag', 'benzene'
    explainers = ['PGE']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                for loss_type in ['ib', ]:  # 'ibconf1', 'ibconf3'
                    for reg_entropy_loss_weight in [0.3]:
                        print('reg_entropy_loss_weight: ', reg_entropy_loss_weight)
                        evaluator = ConfidenceEvaluator(d, m, e, loss_type=loss_type, force_run=False,
                                                        save_confidence_scores=True)
                        evaluator.explainer_manager.explainer.conf_loss_weight = 100
                        evaluator.explainer_manager.explainer.epochs = 100
                        evaluator.explainer_manager.explainer.reg_coefs = [0.0003, reg_entropy_loss_weight]
                        evaluator.set_experiments()
                        # evaluator.evaluating()
                        # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                        evaluator.show_results()
                        # evaluator.dataset_loader.plot_expl()


if __name__ == '__main__':
    # run_baselines()
    run_time_analyze()
    run_ensembles()
    # run_confpge()
    # survey_the_extreme_edge_value()
    # run_ablation()
    # run_baselines()
    # run_pge_noisy_test()
    # run_confpge_noisy_test()

