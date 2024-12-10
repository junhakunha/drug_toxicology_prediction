import os
import sys
import argparse
import time
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.constants import HOME_DIR, DATA_DIR, CHKPT_DIR, EMBED_DIR
from src.utils.data_prep import smiles_to_graph_RCGN, get_data, SMILESDataset
from src.model.GCN import RGCNRegressionModel
from src.model.LLM import RobertaWithEmbeddings


# load the saved model

def main(args):
    model_chkpt_path = args.model

    tokenizer = RobertaTokenizer.from_pretrained(model_chkpt_path)
    model = RobertaWithEmbeddings.from_pretrained(model_chkpt_path, num_labels=1)  # For regression

    chkpt_time = model_chkpt_path.split('/')[-1].split('_')[1]
    savepath = os.path.join(EMBED_DIR, f'LLM_embeddings_{chkpt_time}')
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

        val_dataset = SMILESDataset(x, y, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        all_embeddings = []
        all_labels = []
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True) # cls_embeddings
            all_embeddings.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        np.save(os.path.join(savepath, f'{split}_x.npy'), all_embeddings)
        np.save(os.path.join(savepath, f'{split}_y.npy'), all_labels)



if __name__ == '__main__':
    model_chkpt_path = CHKPT_DIR + 'LLM_20241210-004824_epoch_11'

    parser = argparse.ArgumentParser(description='Extract embeddings from a trained ChemBERTa model')
    parser.add_argument('--model', type=str, default=model_chkpt_path, help='Path to the saved ChemBERTa model checkpoint')
    args = parser.parse_args()
    
    main(args)
