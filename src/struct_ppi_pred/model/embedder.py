import os
import json
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from struct_ppi_pred.utils.logger import setup_logger
from struct_ppi_pred.model.utils import get_protein_data_loader
from struct_ppi_pred.model.MAPE_PPI.mape_enc import CodeBook

logger = setup_logger() 

class Embedder:
    """
    Generates, saves, and loads protein embeddings using a pre-trained VAE model.

    This class handles the process of creating or loading protein embeddings. It utilizes a pre-trained
    Variational Autoencoder (VAE) model to generate embeddings for protein graphs. The embeddings are either
    generated and saved or loaded from existing files if they are already present.

    Attributes:
        data_path (str): Path to the directory containing the dataset and other related files.
        train_data_path (str): Path to the CSV file containing the training dataset.
        val_data_path (str): Path to the CSV file containing the validation dataset.
        test_data_path (str): Path to the CSV file containing the test dataset.
        processed_data_dir (str): Directory containing processed protein data files.
        embedding_dir (str): Directory where embeddings are saved and loaded from.
        mape_weights_path (str): Path to the pre-trained weights of the MAPE-PPI model.
        device (torch.device): The device (CPU or CUDA) on which to perform computations.
        batch_size (int): Batch size used for generating embeddings.
        vae_model (CodeBook): The loaded VAE model used for generating embeddings.
    """
    def __init__(self,
                data_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data",
                mape_weights_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data/data_sources/vae_model.ckpt",
                batch_size: int = 256,
                ):
        """
        Initializes the Embedder.

        Args:
            data_path (str): Path to the data directory. Defaults to "/path/to/your/data".
            mape_weights_path (str): Path to the pre-trained MAPE-PPI model weights. Defaults to "/path/to/your/vae_model.ckpt".
            batch_size (int): Batch size for embedding generation. Defaults to 256.
            mape_cfg_file (str): Configuration dictionary for the MAPE-PPI model.
        """

        self.data_path = data_path
        self.train_data_path = os.path.join(self.data_path, "train.csv")
        self.val_data_path = os.path.join(self.data_path, "val.csv")
        self.test_data_path = os.path.join(self.data_path, "test.csv")
        self.processed_data_dir = os.path.join(self.data_path, "processed_data")
        self.embedding_dir = os.path.join(self.data_path, "embeddings")
        self.mape_weights_path = mape_weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.mape_cfg_file = os.path.join(Path(__file__).parent.parent, "model/MAPE_PPI/mape_cfg.json")

        with open(self.mape_cfg_file, "r") as f:
            self.mape_cfg = json.load(f)

        # Create embeddings directory if it does not exist
        os.makedirs(self.embedding_dir, exist_ok=True)

        # Initialize the VAE model
        self.vae_model = CodeBook(self.mape_cfg)
        self.vae_model.load_state_dict(torch.load(self.mape_weights_path))
        self.vae_model = self.vae_model.to(self.device)

        # Freeze the VAE model
        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.vae_model.eval()

    def get_protein_list(self):
        """
        Extracts a unique list of protein identifiers from the train, validation, and test datasets.

        Returns:
            list: A list of unique protein identifiers.
        """
        # Read dataset
        train_df = pd.read_csv(self.train_data_path)
        val_df = pd.read_csv(self.val_data_path)
        test_df = pd.read_csv(self.test_data_path)

        # Concatenate datasets
        dataset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Get unique proteins
        protein_list = dataset_df["P1"].unique().tolist()
        protein_list.extend(dataset_df["P2"].unique().tolist())

        protein_list = list(set(protein_list))

        return protein_list

    def run(self):
        """
        Executes the embedding generation or loading process for all unique proteins in the dataset.

        This method orchestrates the main workflow of the Embedder. It first retrieves the list of unique proteins,
        then sets up a DataLoader, and finally generates or loads embeddings in batches.
        """
        protein_list = self.get_protein_list()

        logger.info("Number of unique proteins: %d", len(protein_list))
        logger.info("Device: %s", self.device)

        # Create a data loader for proteins
        data_loader = get_protein_data_loader(
            protein_list,
            self.processed_data_dir,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle for embedding generation
            num_workers=4,
            pin_memory=True
        )

        # Generate/load embeddings
        self.generate_embeddings(data_loader)

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
                    embedding_file_path = os.path.join(self.embedding_dir, f"{protein_id}.pt")
                    if not os.path.exists(embedding_file_path):
                        torch.save(embedding, embedding_file_path)
