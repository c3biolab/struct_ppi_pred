import os
import typer
from pathlib import Path

from struct_ppi_pred.model.embedder import Embedder
from struct_ppi_pred.utils.logger import setup_logger

logger = setup_logger()

embedder_cli = typer.Typer()

@embedder_cli.command("dev")
def generate(
    mape_weights_path: str = typer.Option(help="MAPE weights path"),
    batch_size: int = typer.Option(256, help="Batch size for embedding generation"),
    ):
    """
    Generate or load embeddings for all unique proteins in the dataset.
    """
    logger.info("Starting embedding generation process...")

    embedder = Embedder(data_path=os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data"),
                        mape_weights_path=mape_weights_path, 
                        batch_size=batch_size, 
                        mode="dev")
    embedder.run()

    logger.info("Embeddings generated and saved successfully!")

@embedder_cli.command("inf")
def inference(
    data_path: str = typer.Option(help="Path for inference data"),
    mape_weights_path: str = typer.Option(help="MAPE weights path"),
    batch_size: int = typer.Option(256, help="Batch size for embedding generation"),
    ):
    """
    Generate or load embeddings for all unique proteins in the dataset.
    """
    logger.info("Starting embedding generation process...")

    embedder = Embedder(data_path=data_path, mape_weights_path=mape_weights_path, batch_size=batch_size, mode="inf")
    embedder.run()

    logger.info("Embeddings generated and saved successfully!")
