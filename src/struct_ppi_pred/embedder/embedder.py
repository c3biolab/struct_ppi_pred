import os
import json
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from struct_ppi_pred.utils.logger import setup_logger
from struct_ppi_pred.model.utils import get_protein_data_loader
from struct_ppi_pred.embedder.MAPE_PPI.mape_enc import CodeBook

logger = setup_logger() 

class Embedder:
    def __init__(self,
                protein_list,
                outdir,
                processed_data_dir: str,
                batch_size: int = 256,
                ):
        
        self.protein_list = protein_list
        self.outdir = outdir
        self.processed_data_dir = processed_data_dir
        self.mape_weights_path = os.path.join(Path(__file__).parent.parent.parent.parent, "config/vae_model.ckpt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.mape_cfg_file = os.path.join(Path(__file__).parent.parent.parent.parent, "config/mape_cfg.json")

        with open(self.mape_cfg_file, "r") as f:
            self.mape_cfg = json.load(f)

        # Initialize the VAE model
        self.vae_model = CodeBook(self.mape_cfg)
        self.vae_model.load_state_dict(torch.load(self.mape_weights_path))
        self.vae_model = self.vae_model.to(self.device)

        # Freeze the VAE model
        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.vae_model.eval()

    def generate_embeddings(self, data_loader: DataLoader):
        """
        Generates or loads embeddings for proteins using the provided DataLoader.

        This method iterates through the DataLoader, processing batches of protein graphs. For each protein,
        it either generates a new embedding using the VAE model or loads an existing embedding from the file system
        if it's already present.

        Args:
            data_loader (DataLoader): DataLoader for the protein dataset.
        """
        with torch.no_grad():
            for protein_ids, protein_graphs in tqdm(data_loader, desc="Generating Embeddings"):
                protein_graphs = protein_graphs.to(self.device)

                # Generate embeddings
                embeddings = self.vae_model.Protein_Encoder.forward(protein_graphs, self.vae_model.vq_layer).cpu()

                # Save embeddings
                for protein_id, embedding in zip(protein_ids, embeddings):
                    embedding_file_path = os.path.join(self.outdir, f"{protein_id}_embedding.pt")
                    if not os.path.exists(embedding_file_path):
                        torch.save(embedding, embedding_file_path)

    def run(self):
        """
        Executes the embedding generation or loading process for all unique proteins in the dataset.

        This method orchestrates the main workflow of the Embedder. It first retrieves the list of unique proteins,
        then sets up a DataLoader, and finally generates or loads embeddings in batches.
        """

        logger.info("Number of unique proteins: %d", len(self.protein_list))
        logger.info("Device: %s", self.device)

        # Create a data loader for proteins
        data_loader = get_protein_data_loader(
            self.protein_list,
            self.processed_data_dir,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle for embedding generation
            num_workers=4,
            pin_memory=True
        )

        # Generate/load embeddings
        self.generate_embeddings(data_loader)

    