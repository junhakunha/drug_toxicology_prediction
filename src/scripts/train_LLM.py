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

from src.utils.constants import HOME_DIR, DATA_DIR, CHKPT_DIR
from src.utils.data_prep import smiles_to_graph_RCGN, get_data, SMILESDataset
from src.model.GCN import RGCNRegressionModel
from src.model.LLM import RobertaWithEmbeddings


# Load the pretrained ChemBERTa tokenizer and model

def main():
    # Get pretrained ChemBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = RobertaWithEmbeddings.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=1)  # For regression
    
    # Get data
    data = get_data()
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    test_x, test_y = data['test_x'], data['test_y']

    train_dataset = SMILESDataset(train_x, train_y, tokenizer)
    val_dataset = SMILESDataset(val_x, val_y, tokenizer)
    test_dataset = SMILESDataset(test_x, test_y, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    # Training loop
    # Draw a graph of training loss, validation loss
    # Save the best model and tokenizer based on validation loss
    epochs = 20
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    min_val_epoch = 0
    min_val_model = None
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs[0].squeeze()  # Output shape: [batch_size]

            # Compute loss
            loss = loss_fn(predictions, labels)
            total_train_loss += loss.item()

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        total_train_loss /= len(train_loader)
        train_losses.append(total_train_loss)

        # Validation
        model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                predictions = outputs[0].squeeze() 

                # Compute loss
                loss = loss_fn(predictions, labels)
                total_val_loss += loss.item()
        
        total_val_loss /= len(val_loader)
        val_losses.append(total_val_loss)

        # Save best model based on validation loss
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            min_val_epoch = epoch
            min_val_model = model.state_dict()

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot training loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    # Save both model and tokenizer (save the best model based on validation loss)
    cur_time = time.strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(CHKPT_DIR, f'LLM_{cur_time}_epoch_{min_val_epoch}')

    # Reinitialize the model to load the best model
    model = RobertaWithEmbeddings.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=1)
    model.to(device)
    model.load_state_dict(min_val_model)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    main()