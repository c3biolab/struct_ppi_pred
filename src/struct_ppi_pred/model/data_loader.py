import os
import pickle
import numpy as np
import torch
import dgl
from torch.utils.data import Dataset

class PPIDataset(Dataset):
    """
    Dataset class for Protein-Protein Interaction (PPI) data.

    Attributes:
        dataDF (pd.DataFrame): DataFrame containing PPI pairs and their labels.
        processed_data_dir (str): Directory containing processed protein graph data.
    """
    def __init__(self, dataDF, processed_data_dir):
        """
        Initialize the dataset.

        Args:
            dataDF (pd.DataFrame): DataFrame with columns "P1", "P2", and "Label".
            processed_data_dir (str): Directory containing preprocessed graph and feature data.
        """
        self.dataDF = dataDF
        self.processed_data_dir = processed_data_dir

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataDF)

    def obtain_seq(self,prot_node):
        """
        Generate sequential edges for a protein graph based on node indices.

        Args:
            prot_node (torch.Tensor): Tensor containing node features.

        Returns:
            list: List of tuples representing sequential edges.
        """
        prot_seq = []
        for j in range(prot_node.shape[0]-1):
            prot_seq.append((j, j+1))
            prot_seq.append((j+1, j))
        return prot_seq

    def __getitem__(self, idx):
        """
        Get the data for a single PPI sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing two DGL graphs (protein 1 and protein 2) and the interaction label.
        """
        # Get sample: 1 PPI (1: inter/0: not inter)
        sample = self.dataDF.iloc[idx, :]
        p1 = sample["P1"]
        p2 = sample["P2"]
        label = sample["Label"]

        if not os.path.isfile(os.path.join(self.processed_data_dir, f"{p1}_graph.pkl")):
            r_contacts_p1 = np.load(os.path.join(self.processed_data_dir, f"{p1}_r_contacts.npy"), allow_pickle=True)
            k_contacts_p1 = np.load(os.path.join(self.processed_data_dir, f"{p1}_k_contacts.npy"), allow_pickle=True)
            x_p1 = torch.load(os.path.join(self.processed_data_dir, f"{p1}_nodes.pt"))

            seq_p1 = self.obtain_seq(x_p1)

            prot_g1 = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : seq_p1, 
                                    ('amino_acid', 'STR_KNN', 'amino_acid') : k_contacts_p1.tolist(),
                                    ('amino_acid', 'STR_DIS', 'amino_acid') : r_contacts_p1.tolist()})
            prot_g1.ndata['x'] = torch.FloatTensor(x_p1)

            # Save graph
            with open(os.path.join(self.processed_data_dir, f"{p1}_graph.pkl"), "wb") as tf:
                pickle.dump(prot_g1, tf)

        else:
            with open(os.path.join(self.processed_data_dir, f"{p1}_graph.pkl"), "rb") as tf:
                prot_g1 = pickle.load(tf)

        if not os.path.isfile(os.path.join(self.processed_data_dir, f"{p2}_graph.pkl")):

            r_contacts_p2 = np.load(os.path.join(self.processed_data_dir, f"{p2}_r_contacts.npy"), allow_pickle=True)
            k_contacts_p2 = np.load(os.path.join(self.processed_data_dir, f"{p2}_k_contacts.npy"), allow_pickle=True)
            x_p2 = torch.load(os.path.join(self.processed_data_dir, f"{p2}_nodes.pt"))

            seq_p2 = self.obtain_seq(x_p2)

            prot_g2 = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : seq_p2,
                                        ('amino_acid', 'STR_KNN', 'amino_acid') : k_contacts_p2.tolist(),
                                        ('amino_acid', 'STR_DIS', 'amino_acid') : r_contacts_p2.tolist()})
            
            prot_g2.ndata['x'] = torch.FloatTensor(x_p2)

            # Save graph
            with open(os.path.join(self.processed_data_dir, f"{p2}_graph.pkl"), "wb") as tf:
                pickle.dump(prot_g2, tf)

        else:
            with open(os.path.join(self.processed_data_dir, f"{p2}_graph.pkl"), "rb") as tf:
                prot_g2 = pickle.load(tf)

        return prot_g1, prot_g2, torch.tensor(label, dtype=torch.float32)