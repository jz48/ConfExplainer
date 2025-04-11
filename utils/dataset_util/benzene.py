import os
import torch

from utils.dataset_util.extract_google_datasets import load_graphs

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
]

# benzene_data_dir = os.path.join(os.path.dirname(__file__), 'benzene_data')
# benzene_smiles_df = 'benzene_smiles.csv'

# benzene_datapath = os.path.join(os.path.dirname(__file__), 'benzene.npz')


class Benzene(object):

    def __init__(
            self,
            split_sizes=(0.7, 0.2, 0.1),
            seed=None,
            data_path: str = None,
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.graphs, self.explanations, self.zinc_ids = load_graphs(data_path)
