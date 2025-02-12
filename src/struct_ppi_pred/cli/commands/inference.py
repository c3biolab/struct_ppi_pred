import typer
from struct_ppi_pred.inference.inference import Inference, ProteoformAnalyzer

inference_cli: typer.Typer = typer.Typer()

@inference_cli.command()
def inference(
    data_path: str = typer.Option(help="Path for inference data"),
    human_uniprot_fts_dir: str = typer.Option(help="Path for human uniprot features"),
    bac_uniprot_fts_dir: str = typer.Option(help="Path for bacterial uniprot features"),
    pred_dir: str = typer.Option(help="Path for prediction files"),
    threshold: float = typer.Option(help="Threshold for converting probabilities to binary predictions")
):
    typer.echo("Starting inference process...")
    inference = Inference(
        data_path=data_path,
        human_uniprot_fts_dir=human_uniprot_fts_dir,
        bac_uniprot_fts_dir=bac_uniprot_fts_dir,
        pred_dir_name=pred_dir,
        threshold=threshold
    )
    inference.run()
    inference.aggrerate()

    ProteoformAnalyzer(
        human_uniprot_fts_dir=inference.human_uniprot_fts_dir,
        bac_uniprot_fts_dir=inference.bac_uniprot_fts_dir,
        per_prot_dir=inference.per_prot_dir,
        out_dir=inference.out_dir
    ).proteoform_analysis()
