import torch
import numpy as np
from utils.training import GNNTrainer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def train_(dataset, model):
    trainer = GNNTrainer(dataset, model)
    # trainer.plot()
    trainer.train()


# train_('bareg1', 'GraphGCN')
# train_('bareg2', 'GraphGCN')
train_('bareg3', 'GraphGCN')
# train_('bareg1', 'RegGCN')
# train_('bareg2', 'RegGCN')
# train_('bareg3', 'RegGCN')
# train_('ba2motif', 'GraphGCN')
# train_('ba2-2motifs', 'GraphGCN')
# train_('crippen', 'GraphGCN')
# train_('mutag', 'GraphGCN')
# train_('benzene', 'GraphGCN')
# train_('flca', 'GraphGCN')
# train_('alca', 'GraphGCN')
# train_('triangles_small', 'GraphGCN')

# train_('crippen', 'GAT')
# train_('bareg1', 'GAT')
# train_('bareg2', 'GAT')
# train_('bareg3', 'GAT')
# train_('triangles', 'GAT')

# train_('syn1', 'NodeGCN')
# train_('syn2', 'NodeGCN')
# train_('syn3', 'NodeGCN')
# train_('syn4', 'NodeGCN')
