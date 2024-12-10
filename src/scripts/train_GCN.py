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
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.constants import HOME_DIR, DATA_DIR, CHKPT_DIR, IMAGE_DIR
from src.utils.data_prep import smiles_to_graph_RCGN, get_data
from src.model.GCN import RGCNRegressionModel



def main():
    data = get_data()
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    test_x, test_y = data['test_x'], data['test_y']

    # Convert to PyTorch Geometric Data objects
    train_data = smiles_to_graph_RCGN(train_x, train_y)
    val_data = smiles_to_graph_RCGN(val_x, val_y)
    test_data = smiles_to_graph_RCGN(test_x, test_y)

    # Set random seed for reproducibility
    torch.manual_seed(0)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCNRegressionModel(input_dim=13, num_edge_types=4).to(device)  # Adjust input_dim based on node features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Training loop with eval on validation set
    for epoch in tqdm(range(100)):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch)

            loss = criterion(out.squeeze(), batch.y)

            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        train_losses.append(total_loss)

        # Evaluate on validation set (save best model based on validation loss)
        model.eval()
        min_val_loss = np.inf
        min_val_epoch = 0
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out.squeeze(), batch.y)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                best_model = model.state_dict()


        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot training loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    cur_time = time.strftime('%Y%m%d-%H%M%S')

    # Save image
    plt.savefig(os.path.join(IMAGE_DIR, f'GNN_loss_{cur_time}.png'))

    # Store the model
    torch.save(best_model, os.path.join(CHKPT_DIR, f'rgcn_model_{cur_time}_epoch_{min_val_epoch}.pt'))

    plt.show()

if __name__ == '__main__':
    main()