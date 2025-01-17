import typer
from .commands.train import trainer_cli

cli: typer.Typer = typer.Typer()

@cli.callback()
def main():
    """
    PPI CLI for managing training and evaluation tasks.
    """
    typer.echo("Welcome to the PPI CLI.")

# Add the trainer group
cli.add_typer(trainer_cli, name="trainer")

if __name__ == "__main__":
    cli()
