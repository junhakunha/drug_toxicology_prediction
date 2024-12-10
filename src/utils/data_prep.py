import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pandas as pd

from rdkit import Chem

from src.utils.constants import DATA_DIR, EMBED_DIR


def get_data():
    train_data_path = DATA_DIR + 'splits/train.csv'
    val_data_path = DATA_DIR + 'splits/val.csv'
    test_data_path = DATA_DIR + 'splits/test.csv'

    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)

    train_x = train_data['Canonical_SMILES']
    train_y = train_data['LD50_mgkg'].to_numpy()
    val_x = val_data['Canonical_SMILES']
    val_y = val_data['LD50_mgkg'].to_numpy()
    test_x = test_data['Canonical_SMILES']
    test_y = test_data['LD50_mgkg'].to_numpy()

    y_train = list(np.log(train_y))
    y_val = list(np.log(val_y))
    y_test = list(np.log(test_y))

    data = {
        'train_x': train_x,
        'train_y': y_train,
        'val_x': val_x,
        'val_y': y_val,
        'test_x': test_x,
        'test_y': y_test
    }
    return data


def smiles_to_graph_RCGN(smiles_list, labels):
    """
    Converts a list of SMILES strings into a list of PyTorch Geometric Data objects,
    including bond types as edge attributes. This is for the RGCNRegressionModel.
    """
    def get_atom_hash(atomic_number):
        """ 
        A helper function to quickly encode atomic number into a one-hot vector
        """
        atomic_number_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53] # Atomic numbers of atoms in the dataset (length 13)
        if atomic_number in atomic_number_list:
            return atomic_number_list.index(atomic_number)
        else:
            raise ValueError(f'Atomic number {atomic_number} not supported')

    def get_bond_type(bond):
        """
        Encodes bond type as an integer:
        - Single: 0
        - Double: 1
        - Triple: 2
        - Aromatic: 3
        """
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            return 2
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            return 3
        else:
            raise ValueError(f'Unsupported bond type: {bond_type}')

    data_list = []
    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Add explicit hydrogens to the molecule
        if mol is None:
            raise ValueError(f'Could not parse SMILES string: {smiles}')

        # Node features: One-hot vector depending on the atomic number (the position of the 1 in the vector)
        node_features = []
        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            one_hot = [0] * len([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]) # Length of the atomic number list
            one_hot[get_atom_hash(atomic_number)] = 1
            node_features.append(one_hot)

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge indices and edge types (bond types)
        edge_indices = []
        edge_types = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add edges (both directions for undirected graph)
            edge_indices.append((i, j))
            edge_indices.append((j, i))
            
            # Add bond type for both directions
            bond_type = get_bond_type(bond)
            edge_types.append(bond_type)
            edge_types.append(bond_type)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Target value (LD50 label)
        y = torch.tensor([label], dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
        data_list.append(data)

    return data_list


def smiles_to_graph_GCN(smiles_list, labels):
    """
    Converts a list of SMILES strings into a list of PyTorch Geometric Data objects.
    This is for the GCNRegressionModel.
    """
    def get_atom_hash(atomic_number):
        """ 
        A helper function to quickly encode atomic number into a one-hot vector
        """
        atomic_number_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53] # Atomic numbers of atoms in the dataset (length 13)
        if atomic_number in atomic_number_list:
            return atomic_number_list.index(atomic_number)
        else:
            raise ValueError(f'Atomic number {atomic_number} not supported')

    data_list = []
    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            raise ValueError(f'Could not parse SMILES string: {smiles}')

        # Node features: One-hot vector depending on the atomic number (the position of the 1 in the vector)
        node_features = []
        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            one_hot = [0] * 13
            one_hot[get_atom_hash(atomic_number)] = 1
            node_features.append(one_hot)

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge indices and edge attributes
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append((i, j))
            edge_indices.append((j, i))  # Undirected graph
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Target value
        y = torch.tensor([label], dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        # Tokenize the SMILES string
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return input IDs, attention mask, and label
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }
    

def load_embeddings(GNN_embed_path, LLM_embed_path):
    """
    Load embeddings from a directory containing .npy files
    """
    gcn_embedding_path = os.path.join(EMBED_DIR, GNN_embed_path)
    llm_embedding_path = os.path.join(EMBED_DIR, LLM_embed_path)

    gcn_train_x = np.load(os.path.join(gcn_embedding_path, 'train_x.npy'))
    gcn_val_x = np.load(os.path.join(gcn_embedding_path, 'val_x.npy'))
    gcn_test_x = np.load(os.path.join(gcn_embedding_path, 'test_x.npy'))

    gcn_train_y = np.load(os.path.join(gcn_embedding_path, 'train_y.npy'))
    gcn_val_y = np.load(os.path.join(gcn_embedding_path, 'val_y.npy'))
    gcn_test_y = np.load(os.path.join(gcn_embedding_path, 'test_y.npy'))

    llm_train_x = np.load(os.path.join(llm_embedding_path, 'train_x.npy'))
    llm_val_x = np.load(os.path.join(llm_embedding_path, 'val_x.npy'))
    llm_test_x = np.load(os.path.join(llm_embedding_path, 'test_x.npy'))

    llm_train_y = np.load(os.path.join(llm_embedding_path, 'train_y.npy'))
    llm_val_y = np.load(os.path.join(llm_embedding_path, 'val_y.npy'))
    llm_test_y = np.load(os.path.join(llm_embedding_path, 'test_y.npy'))



    # Verify that the labels are the same
    assert np.allclose(gcn_train_y, llm_train_y)
    assert np.allclose(gcn_val_y, llm_val_y)
    assert np.allclose(gcn_test_y, llm_test_y)

    train_x = np.concatenate([gcn_train_x, llm_train_x], axis=1)
    val_x = np.concatenate([gcn_val_x, llm_val_x], axis=1)
    test_x = np.concatenate([gcn_test_x, llm_test_x], axis=1)

    data = {
        'train_x': torch.tensor(train_x, dtype=torch.float),
        'train_y': torch.tensor(gcn_train_y, dtype=torch.float),
        'val_x': torch.tensor(val_x, dtype=torch.float),
        'val_y': torch.tensor(gcn_val_y, dtype=torch.float),
        'test_x': torch.tensor(test_x, dtype=torch.float),
        'test_y': torch.tensor(gcn_test_y, dtype=torch.float),
        'GCN_index': gcn_train_x.shape[1]
    }
    return data