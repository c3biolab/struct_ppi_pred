import pandas as pd
import torch
import dgl
from torch.utils.data import DataLoader

from .data_loader import PPIDataset

def collate(samples):
    """
    Custom collate function for batching PPI data.

    This function separates graphs and labels from the samples, batches the graphs, and stacks the labels.

    Args:
        samples (list): List of tuples where each tuple contains two DGL graphs (protein 1 and protein 2) and a label.

    Returns:
        tuple:
            - dgl.DGLGraph: Batched graph for protein 1.
            - dgl.DGLGraph: Batched graph for protein 2.
            - torch.Tensor: Tensor of labels.
    """
    # Separate the graphs and labels
    graphs1, graphs2, labels = zip(*samples)
    # Batch the graphs
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    # Stack the labels into a single tensor
    labels = torch.stack(labels, dim=0)
    return batched_graph1, batched_graph2, labels

def get_data_loader(data_path, processed_data_dir, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
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
        collate_fn=collate, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return data_loader
