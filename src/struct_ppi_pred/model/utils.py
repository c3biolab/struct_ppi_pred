import pandas as pd
import torch
import dgl
from torch.utils.data import DataLoader

from .data_loader import PPIDataset

def collate(samples):
    # Separate the graphs and labels
    graphs1, graphs2, labels = zip(*samples)
    # Batch the graphs
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    # Stack the labels into a single tensor
    labels = torch.stack(labels, dim=0)
    return batched_graph1, batched_graph2, labels

def get_data_loader(data_path, processed_data_dir, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    dataDF = pd.read_csv(data_path)
    dataset = PPIDataset(dataDF, processed_data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader