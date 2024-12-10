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

    data = get_data()
    val_x, val_y = data['val_x'], data['val_y']

    # Convert to PyTorch Geometric Data objects
    val_data = smiles_to_graph_RCGN(val_x, val_y)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCNRegressionModel(input_dim=13, num_edge_types=4).to(device)
    model.load_state_dict(torch.load(model_chkpt_path))

    # Extract embeddings of validation set
    all_embeddings = []
    for batch in val_loader:
        batch = batch.to(device)
        embeddings = model(batch, return_embeds=True)
        print(embeddings.shape)
        all_embeddings.append(embeddings.detach().cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    print(all_embeddings.shape)
    cur_time = time.strftime('%Y%m%d-%H%M%S')
    np.save(EMBED_DIR + f'val_embeddings_{cur_time}.npy', all_embeddings)


if __name__ == '__main__':
    model_chkpt_path = CHKPT_DIR + 'rgcn_model_20241209-193100.pt'

    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default=model_chkpt_path)
    args = args.parse_args()
    main(args)