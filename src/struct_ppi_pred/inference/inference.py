import os
import json
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from struct_ppi_pred.utils.logger import setup_logger
from struct_ppi_pred.model.ppi_model import PPI_Model

logger = setup_logger() 

class ProteinPairsDataset(Dataset):
    def __init__(self, p1, pool_B, processed_data_dir):
        """
        Initialize the dataset.

        Args:
            p1 (str): The id of a protein.
            pool_B (list): A list of ids of proteins in the pool to be paired with p1.
            processed_data_dir (str): The directory containing preprocessed protein data files.
        """
        self.p1 = p1
        self.processed_data_dir = processed_data_dir
        self.p1_embedding = torch.load(os.path.join(self.processed_data_dir, f"{p1}_embedding.pt"))
        self.pool_B = pool_B

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.pool_B)

    def __getitem__(self, idx):
        """
        Get the embeddings for a given protein pair.

        Args:
            idx (int): The index of the protein pair in the pool.

        Returns:
            tuple: A tuple containing the embeddings of the two proteins.
        """
        sample = self.pool_B[idx]

        # Load embeddings
        prot_g2 = torch.load(os.path.join(self.processed_data_dir, f"{sample}_embedding.pt"))

        return self.p1_embedding, prot_g2

class Inference():
    """
    Class for running inference on a dataset of protein pairs.

    Attributes:
        data_path (str): Path to the data directory.
        batch_size (int): Batch size to use for inference.
        output_dir (str): Directory where the inference results should be saved.
        pred_dir_name (str): The name of the subdirectory where the inference results should be saved.
        threshold (float): Threshold for converting probabilities to binary predictions.
    """
    def __init__(self,
                data_path="/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data/gut_data",
                batch_size: int = 256,
                output_dir: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/output",
                pred_dir_name: str = "Healthy",
                threshold = None
                ):
        
        """
        Initialize the Inference class.

        Args:
            data_path (str): Path to the data directory. Defaults to "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data/gut_data".
            batch_size (int): Batch size to use for inference. Defaults to 256.
            output_dir (str): Directory where the inference results should be saved. Defaults to "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/output".
            pred_dir_name (str): The name of the subdirectory where the inference results should be saved. Defaults to "Healthy".
            threshold (float): The threshold for converting model probabilities to binary predictions. Must be provided.
        """
        self.best_model_path = os.path.join(output_dir, "best_model.pt")
        self.out_dir = os.path.join(output_dir, pred_dir_name)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.data_path = data_path
        self.batch_size = batch_size
        self.pair_info_file = os.path.join(self.data_path, "human_gut_pairs_Healthy.json")
        self.processed_data_dir = os.path.join(self.data_path, "processed_data")
        self.per_prot_dir = os.path.join(self.out_dir, "PerProtein")
        
        os.makedirs(self.per_prot_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        if self.threshold is None:
            raise ValueError("Threshold must be provided")

    def get_protein_pools(self):
        with open(self.pair_info_file, "r") as f:
            pair_info = json.load(f)

        pool_A = pair_info["Pool_A"]
        pool_B = pair_info["Pool_B"]

        return pool_A, pool_B

    def get_ppi_data_loader(self, p1, pool_B, shuffle=False, num_workers=12, pin_memory=True):  
        """
        Creates a DataLoader for a protein pair dataset.

        Args:
            p1 (str): The id of a protein.
            pool_B (list): A list of ids of proteins in the pool to be paired with p1.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to False.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 12.
            pin_memory (bool, optional): Whether to use pinned memory for faster data transfer to GPU. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the protein pair dataset.
        """
        dataset = ProteinPairsDataset(p1, pool_B, self.processed_data_dir)
        # Create the DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
        )
        return data_loader

    def run(self):
        """
        Runs inference for each protein in pool_A with all proteins in pool_B,
        saving the predicted interactors to a JSON file for each protein.

        The JSON file is saved in the directory specified by `self.per_prot_dir`
        and is named after the protein id, with a suffix of "_predictions.json".

        The file contains a dictionary with keys as the indices of the interactors
        and values as dictionaries with two keys: "bacterial_protein" and "score".
        The value of "bacterial_protein" is the id of the bacterial protein that
        interacts with the human protein, and the value of "score" is the predicted
        probability of interaction.

        The function does not return anything, but saves the results to disk.
        """
        pool_A, pool_B = self.get_protein_pools()

        # Load model
        model = PPI_Model()
        model.load_state_dict(torch.load(self.best_model_path))
        model.to(self.device)
        model.eval()

        for idx, p1 in enumerate(pool_A):

            if os.path.isfile(os.path.join(self.per_prot_dir,"{}_predictions.json".format(p1))):
                continue

            logger.info(f"Running inference for protein {p1}: {idx+1}/{len(pool_A)}")

            data_loader = self.get_ppi_data_loader(p1, pool_B)

            interactors = {}
            int_idx = 0

            with torch.no_grad():
                for data in tqdm(data_loader):
                    p1_embedding, prot_g2 = data
                    p1_embedding, prot_g2 = p1_embedding.to(self.device), prot_g2.to(self.device)

                    outputs = model(p1_embedding, prot_g2)
                    outputs = torch.sigmoid(outputs)

                    for j in range(outputs.shape[0]):
                        if outputs[j] >= self.threshold:
                            interactors[int_idx] = {
                                "bacterial_protein": pool_B[j],
                                "score": outputs[j].item()
                            }

                            int_idx += 1

                logger.info(f"Found {len(interactors)} interactors for {p1}")

                with open(os.path.join(self.per_prot_dir,"{}_predictions.json".format(p1)), "w") as f:
                    json.dump(interactors, f, indent=4)

    def aggrerate(self):
        """
        Aggregates individual protein interaction predictions into a network file.

        This method iterates over prediction files, collects interactions, and writes them to a network file.
        It also logs the number of proteins involved in interactions and the total number of predictions.

        Attributes:
            net_file (str): Path to the network file where aggregated interactions are saved.

        Steps:
            1. Retrieves protein pools (pool_A and pool_B).
            2. Initializes counters and sets to track involved proteins.
            3. Iterates through prediction files in the specified directory.
            4. For each prediction file, updates the sets of involved proteins and writes interactions to the network file.
            5. Logs the total number of involved proteins and predictions.

        Note:
            - The function assumes prediction files are in JSON format with a specific structure.
            - Predictions with empty content are skipped.
        """
        net_file = os.path.join(self.out_dir, "net.txt")

        with open(net_file, "w") as net_f:
            pool_A, pool_B = self.get_protein_pools()

            tot = 0
            pool_A_involved = set()
            pool_B_involved = set()

            for filename in os.listdir(self.per_prot_dir):
                with open(os.path.join(self.per_prot_dir, filename), "r") as f:
                    prediction = json.load(f)

                    if len(prediction) == 0:
                        continue
                    
                    pool_A_involved.add(filename.split(".")[0].split("_")[0])

                    for _, value in prediction.items():
                        pool_B_involved.add(value["bacterial_protein"])

                        net_f.write("{}\t{}\t{}\n".format(filename.split(".")[0].split("_")[0], value["bacterial_protein"], value["score"]))

                tot += len(prediction)

            logger.info(f"Pool A proteins: {len(pool_A)}")
            logger.info(f"Pool B proteins: {len(pool_B)}")
            logger.info(f"Pool A involved: {len(pool_A_involved)}")
            logger.info(f"Pool B involved: {len(pool_B_involved)}")
            logger.info(f"Total predictions: {tot}")