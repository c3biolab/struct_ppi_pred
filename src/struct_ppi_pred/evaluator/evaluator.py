import os
import numpy as np
import json
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
import torch

from tqdm import tqdm

from struct_ppi_pred.utils.logger import setup_logger
from struct_ppi_pred.model.utils import get_ppi_data_loader
from struct_ppi_pred.model.ppi_model import PPI_Model

logger = setup_logger()

class Evaluator:
    """
    Evaluator class for assessing the performance of a deep learning model,
    particularly for imbalanced protein-protein interaction prediction.
    """

    def __init__(self, 
                 mode: str, 
                 threshold = None,
                 batch_size: int = 256,
                 data_path: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/data",
                 output_dir: str = "/home/c3biolab/c3biolab_projects/doctorals/d/struct_ppi_pred/output"):
        """
        Initializes the Evaluator with a default threshold and output directory.

        Args:
            mode (str): Mode of evaluation (val or test).
            threshold (float): Threshold for converting probabilities to binary predictions.
            output_dir (str): Directory to save results (confusion matrix, plots).
        """
        self.mode = mode
        self.threshold = threshold
        self.batch_size = batch_size
        self.data_path = data_path
        self.output_dir = output_dir
        self.processed_data_dir = os.path.join(self.data_path, "processed_data")
        self.model_path = os.path.join(self.output_dir, "best_model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report_metrics(
        self, true_labels: list, predicted_probs: list, threshold: float = None, only_display = True
    ) -> dict:
        """
        Generate and log a detailed metrics report, save results to files, and return metrics in a dictionary.

        Args:
            true_labels (list): Ground truth labels (0 or 1).
            predicted_probs (list): Model predictions (probabilities of the positive class).
            threshold (float, optional): Threshold for converting probabilities to binary
                                         predictions. Defaults to self.threshold.

        Returns:
            dict: A dictionary containing all calculated metrics.
        """

        if threshold is None:
            threshold = self.threshold

        # Convert probabilities to binary predictions
        predicted_labels = (np.array(predicted_probs) >= threshold).astype(int)

        # Calculate metrics
        metrics = {}
        metrics["precision_macro"] = precision_score(
            true_labels, predicted_labels, average="macro"
        )
        metrics["recall_macro"] = recall_score(
            true_labels, predicted_labels, average="macro"
        )
        metrics["f1_macro"] = f1_score(true_labels, predicted_labels, average="macro")
        metrics["precision_binary"] = precision_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["recall_binary"] = recall_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["f1_binary"] = f1_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["mcc"] = matthews_corrcoef(true_labels, predicted_labels)
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            true_labels, predicted_labels
        )
        metrics["ap"] = average_precision_score(true_labels, predicted_probs)

        try:
            metrics["auroc"] = roc_auc_score(true_labels, predicted_probs)
        except ValueError:
            logger.warning(
                "Only one class present in y_true. ROC AUC score is not defined in that case."
            )
            metrics["auroc"] = np.nan

        precision, recall, pr_thresholds = precision_recall_curve(
            true_labels, predicted_probs
        )
        metrics["recall_at_precision_50"] = recall[np.argmin(np.abs(precision - 0.5))]

        fpr, tpr, roc_thresholds = roc_curve(true_labels, predicted_probs)

        if not only_display:

            # Confusion Matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )

            # Save confusion matrix to CSV
            cm_df.to_csv(os.path.join(self.output_dir, "confusion_matrix.csv"))

            # Plotting using Plotly
            self.plot_pr_curve(precision, recall, metrics["ap"])
            self.plot_roc_curve(fpr, tpr, metrics["auroc"])

            # Print the confusion matrix
            logger.info("Confusion Matrix:\n%s", cm_df)

        # Pretty - Print metrics with the logger
        logger.info("Macro-averaged Metrics:")
        logger.info("  Precision (Macro): %.4f", metrics["precision_macro"])
        logger.info("  Recall (Macro): %.4f", metrics["recall_macro"])
        logger.info("  F1-Score (Macro): %.4f", metrics["f1_macro"])
        logger.info("Interacting Class Metrics:")
        logger.info("  Precision (Interacting): %.4f", metrics["precision_binary"])
        logger.info("  Recall (Interacting): %.4f", metrics["recall_binary"])
        logger.info("  F1-Score (Interacting): %.4f", metrics["f1_binary"])
        logger.info("Imbalance-Aware Metrics:")
        logger.info("  Matthews Correlation Coefficient (MCC): %.4f", metrics["mcc"])
        logger.info("  Balanced Accuracy: %.4f", metrics["balanced_accuracy"])
        logger.info("  Average Precision (AP): %.4f", metrics["ap"])
        logger.info("  Area Under ROC Curve (AUROC): %.4f", metrics["auroc"])
        logger.info("  Recall@Precision=0.5: %.4f", metrics["recall_at_precision_50"])

        return metrics

    def plot_pr_curve(self, precision, recall, ap):
        """
        Plots the Precision-Recall curve using Plotly and saves it as an SVG file.

        Args:
            precision (np.ndarray): Array of precision values.
            recall (np.ndarray): Array of recall values.
            ap (float): Average precision score.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"AP = {ap:.2f}")
        )
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
        )
        fig.write_image(os.path.join(self.output_dir, "pr_curve.svg"))

    def plot_roc_curve(self, fpr, tpr, auroc):
        """
        Plots the ROC curve using Plotly and saves it as an SVG file.

        Args:
            fpr (np.ndarray): Array of false positive rates.
            tpr (np.ndarray): Array of true positive rates.
            auroc (float): Area under the ROC curve.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUROC = {auroc:.2f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
        )
        fig.write_image(os.path.join(self.output_dir, "roc_curve.svg"))

    def obtain_predictions(self):
        """
        Obtains predictions for the validation/test set using the trained model.
        """

        if os.path.isfile(self.prediction_file):
            logger.info("Prediction file already exists. Skipping prediction.")
            return

        logger.info("Obtaining predictions for {} set...".format(self.mode))

        val_loader = get_ppi_data_loader(self.val_data_path, 
                                     self.processed_data_dir,
                                     batch_size=self.batch_size, 
                                     shuffle=False, 
                                     num_workers=4, 
                                     pin_memory=True)

        model = PPI_Model()
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        sample_idx = 0
        pred_dict = {}

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Predicting", leave=False):
                p1, p2, labels = data
                p1, p2, labels = p1.to(self.device), p2.to(self.device), labels.to(self.device)

                outputs = model(p1, p2)
                probs = torch.sigmoid(outputs)
                
                for i in range(len(labels)):
                    pred_dict[sample_idx] = {
                        "label": labels[i].item(),
                        "prediction": probs[i].item()
                    }
                    sample_idx += 1

        with open(self.prediction_file, "w") as f:
            json.dump(pred_dict, f, indent=4)


    def threshold_selection(self):
        """
        Selects the optimal threshold value using the validation set.
        """
        
        # Load predictions
        with open(self.prediction_file, "r") as f:
            pred_dict = json.load(f)

        y_true = list()
        y_pred_probs = list()

        for key, value in pred_dict.items():
            y_true.append(value["label"])
            y_pred_probs.append(value["prediction"])

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

        # Calculate F1-scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Find the index of the maximum F1-score
        idx = np.argmax(f1_scores)

        # Get the threshold, precision, and recall at the maximum F1-score
        selected_threshold = thresholds[idx]
        selected_precision = precision[idx]
        selected_recall = recall[idx]
        max_f1 = f1_scores[idx]

        logger.info("Threshold selection for validation set:")
        logger.info("Threshold: %.4f", selected_threshold)
        logger.info("Precision: %.4f", selected_precision)
        logger.info("Recall: %.4f", selected_recall)
        logger.info("F1-score: %.4f", max_f1)

        self.best_threshold = selected_threshold

        # Interactive plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
        fig.add_trace(
            go.Scatter(
                x=[selected_recall],
                y=[selected_precision],
                mode="markers",
                marker=dict(size=10, color="red"),
                name=f"Selected Threshold: {selected_threshold:.2f}<br>Precision: {selected_precision:.2f}<br>Recall: {selected_recall:.2f}<br>F1: {max_f1:.2f}",
            )
        )
        fig.update_layout(
            title="Interactive Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
        )
        
        # Save the figure as an SVG file
        fig.write_image(os.path.join(self.output_dir, "val_pr_curve_thres.svg"))

        return y_true, y_pred_probs, selected_threshold

    def test_process(self):
        """
        Processes the test set and saves the predictions.
        """
        
        # Load predictions
        with open(self.prediction_file, "r") as f:
            pred_dict = json.load(f)

        y_true = list()
        y_pred_probs = list()

        for key, value in pred_dict.items():
            y_true.append(value["label"])
            y_pred_probs.append(value["prediction"])

        self.report_metrics(y_true, y_pred_probs, self.threshold, only_display=False)

    def run(self):
        """
        Main method to run the evaluator.
        """
        
        if self.mode == "val":
            self.val_data_path = os.path.join(self.data_path, "val.csv")
        elif self.mode == "test":
            self.val_data_path = os.path.join(self.data_path, "test.csv")
            if self.threshold is None:
                raise ValueError("Threshold must be provided for test set.")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.prediction_file = os.path.join(self.output_dir, "{}_predictions.json".format(self.mode))

        self.obtain_predictions()

        if self.mode == "val":
            thresh = self.threshold_selection()
            y_true, y_pred_probs, self.threshold = thresh
            self.report_metrics(y_true, y_pred_probs, self.threshold)
        elif self.mode == "test":
            self.test_process()
            

