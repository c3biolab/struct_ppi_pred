import typer
import json
from pathlib import Path

from struct_ppi_pred.model.embedder import Embedder
from struct_ppi_pred.utils.logger import setup_logger

logger = setup_logger()

embedder_cli = typer.Typer()

@embedder_cli.command()
def generate(
    batch_size: int = typer.Option(256, help="Batch size for embedding generation"),
    ):
    """
    Generate or load embeddings for all unique proteins in the dataset.
    """
    logger.info("Starting embedding generation process...")

    embedder = Embedder(batch_size=batch_size)
    embedder.run()

    logger.info("Embeddings generated and saved successfully!")