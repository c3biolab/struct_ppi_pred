import os
import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from struct_ppi_pred.utils.logger import setup_logger
from struct_ppi_pred.model import get_ppi_data_loader, PPI_Model, FocalLoss

logger = setup_logger()
warnings.filterwarnings("ignore")

class Trainer:
    """
    Trainer class to handle the training and validation of the PPI prediction model.

    Attributes:
        data_path (str): Path to the data directory containing train, val, and processed data.
        train_data_path (str): Path to the training data file.
        val_data_path (str): Path to the validation data file.
        processed_data_dir (str): Directory containing processed protein embeddings.
        mape_weights_path (str): Path to the pretrained MAPE-PPI model weights.
        device (torch.device): Device to use for computation (CPU or CUDA).
        epochs (int): Number of training epochs.
        patience (int): Number of epochs for early stopping patience.
        batch_size (int): Batch size for training and validation.
        save_path (str): Path to save the best model.
    """

    def __init__(self, 
                 data_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data",
                 mape_weights_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data/data_sources/vae_model.ckpt", 
                 epochs: int = 10,
                 batch_size: int = 256, 
                 patience: int = 3, 
                 save_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/output"):
        
        self.data_path = data_path
        self.train_data_path = os.path.join(self.data_path, "train.csv")
        self.val_data_path = os.path.join(self.data_path, "val.csv")
        self.processed_data_dir = os.path.join(self.data_path, "processed_data")
        self.mape_weights_path = mape_weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.save_path = save_path

    def run(self):
        """
        Main method to run training and validation.
        """

        # Check if best_model.pt exists
        if os.path.isfile(os.path.join(self.save_path, "best_model.pt")):
            logger.info("Best model already exists. Skipping training.")
            return

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Load data
        train_loader = get_ppi_data_loader(self.train_data_path, 
                                       self.processed_data_dir,
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=4, 
                                       pin_memory=True)

        val_loader = get_ppi_data_loader(self.val_data_path, 
                                     self.processed_data_dir,
                                     batch_size=self.batch_size, 
                                     shuffle=False, 
                                     num_workers=4, 
                                     pin_memory=True)

        logger.info("Train data points: %d", len(train_loader.dataset))
        logger.info("Val data points: %d", len(val_loader.dataset))
        logger.info("Device: %s", self.device)

        # Load MAPE-PPI configuration
        mape_cfg_path = os.path.join(Path(__file__).parent.parent, "model/MAPE_PPI/mape_cfg.json")
        with open(mape_cfg_path, "r") as f:
            param = json.load(f)

        # Initialize model, loss function, and optimizer
        model = PPI_Model(mape_cfg=param, mape_weights_path=self.mape_weights_path).to(self.device)
        criterion = FocalLoss(gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=param.get("learning_rate", 0.001), weight_decay=param.get("weight_decay", 0.0005))

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            logger.info("Epoch %d/%d", epoch, self.epochs)

            # Training phase
            train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer)
            logger.info("Train Loss: %.4f", train_loss)

            # Validation phase
            val_loss = self.validate(model, val_loader, criterion)
            logger.info("Val Loss: %.4f", val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
                logger.info("Model saved to %s", os.path.join(self.save_path, "best_model.pt"))
            else:
                patience_counter += 1
                logger.info("Early stopping patience counter: %d/%d", patience_counter, self.patience)

            if patience_counter >= self.patience:
                logger.info("Early stopping triggered.")
                break

    def train_one_epoch(self, 
                         model: nn.Module, 
                         loader: torch.utils.data.DataLoader, 
                         criterion: nn.Module, 
                         optimizer: optim.Optimizer) -> float:
        """
        Train the model for one epoch.

        Args:
            model (nn.Module): The model to train.
            loader (torch.utils.data.DataLoader): The data loader for training data.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.

        Returns:
            float: Average training loss for the epoch.
        """
        model.train()
        total_loss = 0.0

        for data in tqdm(loader, desc="Training", leave=False):
            p1, p2, labels = data
            p1, p2, labels = p1.to(self.device), p2.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(p1, p2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            
        return total_loss / len(loader.dataset)

    def validate(self, 
                  model: nn.Module, 
                  loader: torch.utils.data.DataLoader, 
                  criterion: nn.Module) -> float:
        """
        Validate the model on the validation dataset.

        Args:
            model (nn.Module): The model to validate.
            loader (torch.utils.data.DataLoader): The data loader for validation data.
            criterion (nn.Module): The loss function.

        Returns:
            float: Average validation loss.
        """
        model.eval()
        total_loss = 0.0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for data in tqdm(loader, desc="Validating", leave=False):
                p1, p2, labels = data
                p1, p2, labels = p1.to(self.device), p2.to(self.device), labels.to(self.device)

                outputs = model(p1, p2)
                loss = criterion(outputs, labels.unsqueeze(1))

                total_loss += loss.item() * labels.size(0)

                # Store predictions and labels for metrics calculation
                predictions = torch.sigmoid(outputs).cpu() > 0.5  # Threshold at 0.5
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.int().numpy())
                
                # Compute detailed metrics
        self.report_metrics(all_labels, all_predictions)

        return total_loss / len(loader.dataset)

    def report_metrics(self, true_labels: list, predicted_labels: list) -> None:
        """
        Generate and log a detailed metrics report.

        Args:
            true_labels (list): Ground truth labels.
            predicted_labels (list): Model predictions.

        Returns:
            None
        """
        logger.info("Validation Metrics Report:")
        logger.info("\n" + classification_report(true_labels, predicted_labels, target_names=["Non-interacting", "Interacting"]))

        precision_micro = precision_score(true_labels, predicted_labels, average="micro")
        recall_micro = recall_score(true_labels, predicted_labels, average="micro")
        f1_micro = f1_score(true_labels, predicted_labels, average="micro")

        precision_macro = precision_score(true_labels, predicted_labels, average="macro")
        recall_macro = recall_score(true_labels, predicted_labels, average="macro")
        f1_macro = f1_score(true_labels, predicted_labels, average="macro")

        logger.info("Precision (Micro): %.4f", precision_micro)
        logger.info("Recall (Micro): %.4f", recall_micro)
        logger.info("F1-Score (Micro): %.4f", f1_micro)

        logger.info("Precision (Macro): %.4f", precision_macro)
        logger.info("Recall (Macro): %.4f", recall_macro)
        logger.info("F1-Score (Macro): %.4f", f1_macro)

if __name__ == "__main__":
    trainer = Trainer(epochs=50, patience=3)
    trainer.run()
