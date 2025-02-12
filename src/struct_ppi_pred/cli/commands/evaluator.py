import typer
from struct_ppi_pred.evaluator.evaluator import Evaluator

eval_cli: typer.Typer = typer.Typer()

@eval_cli.command("val")
def val_process():
    typer.echo("Starting validation process...")
    evaluator = Evaluator(mode="val")
    evaluator.run()

@eval_cli.command("test")
def test_process(threshold: float = typer.Option(help="Threshold for converting probabilities to binary predictions")):
    evaluator = Evaluator(mode="test", threshold=threshold)
    evaluator.run()
