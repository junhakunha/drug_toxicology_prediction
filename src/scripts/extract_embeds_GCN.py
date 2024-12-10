import os
import sys
import argparse
import time

sys.path.append(os.getcwd())
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.constants import HOME_DIR, DATA_DIR, CHKPT_DIR, EMBED_DIR
from src.utils.data_prep import smiles_to_graph_RCGN, get_data
from src.model.GCN import RGCNRegressionModel


def main(args):
    model_chkpt_path = args.model

    chkpt_time = model_chkpt_path.split('/')[-1].split('_')[2]
    savepath = os.path.join(EMBED_DIR, f'GNN_embeddings_{chkpt_time}')
    os.makedirs(savepath, exist_ok=True)
    

    for split in ['train', 'val', 'test']:
        data = get_data()
        if split == 'train':
            x, y = data['train_x'], data['train_y']
        elif split == 'val':
            x, y = data['val_x'], data['val_y']
        elif split == 'test':
            x, y = data['test_x'], data['test_y']
        else:
            raise ValueError('Invalid split argument. Choose from train/val/test')

        # Convert to PyTorch Geometric Data objects
        split_data = smiles_to_graph_RCGN(x, y)
        dataloader = DataLoader(split_data, batch_size=64, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RGCNRegressionModel(input_dim=13, num_edge_types=4).to(device)
        model.load_state_dict(torch.load(model_chkpt_path))

        # Extract embeddings of validation set
        all_embeddings = []
        all_labels = []
        for batch in dataloader:
            batch = batch.to(device)
            labels = batch.y
            embeddings = model(batch, return_embeds=True)
            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        np.save(os.path.join(savepath, f'{split}_x.npy'), all_embeddings)
        np.save(os.path.join(savepath, f'{split}_y.npy'), all_labels)


if __name__ == '__main__':
    model_chkpt_path = CHKPT_DIR + 'rgcn_model_20241210-043903_epoch_199.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model_chkpt_path)
    args = parser.parse_args()
    main(args)