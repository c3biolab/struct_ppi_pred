import typer
from struct_ppi_pred.inference.inference import Inference, ProteoformAnalyzer

inference_cli: typer.Typer = typer.Typer()

@inference_cli.command()
def start():
    typer.echo("Starting inference process...")
    inference = Inference(threshold=0.99)
    #inference.run()
    #inference.aggrerate()

    ProteoformAnalyzer(
        human_uniprot_fts_dir=inference.human_uniprot_fts_dir,
        bac_uniprot_fts_dir=inference.bac_uniprot_fts_dir,
        per_prot_dir=inference.per_prot_dir,
        out_dir=inference.out_dir
    ).proteoform_analysis()
