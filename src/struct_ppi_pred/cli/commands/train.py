import typer
from struct_ppi_pred.trainer import Trainer

trainer_cli: typer.Typer = typer.Typer()

@trainer_cli.command("start")
def start_training(epochs: int = 100, batch_size: int = 256, patience: int = 3):
    """
    Start training the PPI model.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Early stopping patience.
    """
    typer.echo(f"Starting training with epochs={epochs}, batch_size={batch_size}, patience={patience}")
    trainer = Trainer(epochs=epochs, batch_size=batch_size, patience=patience)
    trainer.run()

@trainer_cli.command("results")
def display_results():
    """
    Display the latest training or validation results.
    """
    typer.echo("Displaying results...")
