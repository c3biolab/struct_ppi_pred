import typer
from struct_ppi_pred.inference.inference import Inference

inference_cli: typer.Typer = typer.Typer()

@inference_cli.command()
def start():
    typer.echo("Starting inference process...")
    inference = Inference(threshold=0.99)
    inference.run()
    inference.aggrerate()
