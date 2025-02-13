import json
import pandas as pd
from struct_ppi_pred.data_handler.protein_data import ProteinDataHandler

class InferenceDataHandler():
    def __init__(self, pairs_file):
        with open(pairs_file, "r") as f:
            self.pairs = json.load(f)
        
        prot_list = list(set(self.pairs["Pool_A"] + self.pairs["Pool_B"]))

        ProteinDataHandler(protein_list=prot_list, session="INFERENCE").run()


InferenceDataHandler(
    pairs_file = "./pairs.json"
)