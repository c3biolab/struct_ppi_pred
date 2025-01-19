import typer
from struct_ppi_pred.cli.commands import trainer_cli, embedder_cli

cli = typer.Typer()

@cli.callback()
def main():
    """
    PPI CLI for managing training, evaluation, and embedding generation tasks.
    """
    typer.echo("Welcome to the PPI CLI.")

# Add the trainer and embedder groups
cli.add_typer(trainer_cli, name="trainer")
cli.add_typer(embedder_cli, name="embedder")

if __name__ == "__main__":
    cli()