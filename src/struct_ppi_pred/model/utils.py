import pandas as pd
import torch
import dgl
from torch.utils.data import DataLoader

from .data_loader import ProteinDataset, PPIDataset

def collate(samples):
    """
    Collates a list of samples into a batch for use in a DataLoader.

    This function is used as the `collate_fn` in a PyTorch DataLoader to handle the batching of protein graphs.
    It takes a list of (protein_id, protein_graph) tuples and returns a tuple containing a list of protein IDs
    and a batched DGL graph.

    Args:
        samples (list): List of tuples, where each tuple contains a protein ID and its corresponding DGL graph.

    Returns:
        tuple: Tuple containing a list of protein IDs and a batched DGL graph representing the batch of proteins.
    """
    protein_ids, graphs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return protein_ids, batched_graph

def get_protein_data_loader(protein_list, processed_data_dir, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    """
    Creates a DataLoader for protein data.

    This function creates a PyTorch DataLoader instance for batching, shuffling, and parallel loading of protein graph data.
    It uses the `ProteinDataset` class to load and prepare the data, and the `collate` function to handle batching.

    Args:
        protein_list (list): List of protein identifiers.
        processed_data_dir (str): Directory containing preprocessed protein data files.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to use pinned memory for faster data transfer to GPU. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the protein dataset.
    """
    dataset = ProteinDataset(protein_list, processed_data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader

def get_ppi_data_loader(data_path, processed_data_dir, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create a data loader for the PPI dataset.

    This function loads the dataset from a CSV file, initializes a `PPIDataset` instance, and returns a PyTorch DataLoader
    for batching, shuffling, and parallel data loading.

    Args:
        data_path (str): Path to the CSV file containing PPI data with columns "P1", "P2", and "Label".
        processed_data_dir (str): Directory containing preprocessed graph and feature data.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to use pinned memory for faster data transfer to GPU. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the PPI dataset.
    """
    # Load the dataset as a DataFrame
    dataDF = pd.read_csv(data_path)

    # Initialize the dataset
    dataset = PPIDataset(dataDF, processed_data_dir)
    # Create the DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return data_loader
