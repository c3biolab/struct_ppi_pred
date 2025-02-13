import os
import requests
import re
import math
import numpy as np
import json
import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path

from struct_ppi_pred.data_handler.protein_data.utils import fetch_protein_data_uniprot, fetch_protein_strcture_af
from struct_ppi_pred.embedder import Embedder
from struct_ppi_pred.utils.logger import setup_logger

logger = setup_logger()

class ProteinDataHandler:
    def __init__(self, protein_list, session):
        self.protein_list = protein_list
        self.session = session
        self.tmp_dir = os.path.join(Path(__file__).parent.parent.parent.parent.parent, f'.tmp_{self.session}')

    def dist(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def read_atoms(self, file, chain="."):
        pattern = re.compile(chain)

        atoms = []
        ajs = []
        
        for line in file:
            line = line.strip()
            if line.startswith("ATOM"):
                type = line[12:16].strip()
                chain = line[21:22]
                if type == "CA" and re.match(pattern, chain):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ajs_id = line[17:20]
                    atoms.append((x, y, z))
                    ajs.append(ajs_id)
                    
        return atoms, ajs
     
    def compute_contacts(self, atoms, threshold):
        contacts = []
        for i in range(len(atoms)-2):
            for j in range(i+2, len(atoms)):
                if self.dist(atoms[i], atoms[j]) < threshold:
                    contacts.append((i, j))
                    contacts.append((j, i))
        return contacts


    def knn(self, atoms, k=5):
        
        x = np.zeros((len(atoms), len(atoms)))
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                x[i, j] = self.dist(atoms[i], atoms[j])
        index = np.argsort(x, axis=-1)
        
        contacts = []
        for i in range(len(atoms)):
            num = 0
            for j in range(len(atoms)):
                if index[i, j] != i and index[i, j] != i-1 and index[i, j] != i+1:
                    contacts.append((i, index[i, j]))
                    num += 1
                if num == k:
                    break
                
        return contacts


    def pdb_to_cm(self, file, threshold):
        atoms, x = self.read_atoms(file)
        r_contacts = self.compute_contacts(atoms, threshold)
        k_contacts = self.knn(atoms)
        return r_contacts, k_contacts, x

    def match_feature(self, x, all_for_assign):
        x_p = np.zeros((len(x), 7))
        
        for j in range(len(x)):
            if x[j] == 'ALA':
                x_p[j] = all_for_assign[0, :]
            elif x[j] == 'CYS':
                x_p[j] = all_for_assign[1, :]
            elif x[j] == 'ASP':
                x_p[j] = all_for_assign[2, :]
            elif x[j] == 'GLU':
                x_p[j] = all_for_assign[3, :]
            elif x[j] == 'PHE':
                x_p[j] = all_for_assign[4, :]
            elif x[j] == 'GLY':
                x_p[j] = all_for_assign[5, :]
            elif x[j] == 'HIS':
                x_p[j] = all_for_assign[6, :]
            elif x[j] == 'ILE':
                x_p[j] = all_for_assign[7, :]
            elif x[j] == 'LYS':
                x_p[j] = all_for_assign[8, :]
            elif x[j] == 'LEU':
                x_p[j] = all_for_assign[9, :]
            elif x[j] == 'MET':
                x_p[j] = all_for_assign[10, :]
            elif x[j] == 'ASN':
                x_p[j] = all_for_assign[11, :]
            elif x[j] == 'PRO':
                x_p[j] = all_for_assign[12, :]
            elif x[j] == 'GLN':
                x_p[j] = all_for_assign[13, :]
            elif x[j] == 'ARG':
                x_p[j] = all_for_assign[14, :]
            elif x[j] == 'SER':
                x_p[j] = all_for_assign[15, :]
            elif x[j] == 'THR':
                x_p[j] = all_for_assign[16, :]
            elif x[j] == 'VAL':
                x_p[j] = all_for_assign[17, :]
            elif x[j] == 'TRP':
                x_p[j] = all_for_assign[18, :]
            elif x[j] == 'TYR':
                x_p[j] = all_for_assign[19, :]
                
        return x_p

    def check_process(self, process_file):
        if os.path.isfile(process_file):
            with open(process_file, "r") as f:
                ready_proteins = json.load(f)
                logger.info(f'{len(ready_proteins)} proteins already processed')
        else:
            ready_proteins = list()

        to_process = list(set(self.protein_list) - set(ready_proteins))
        return ready_proteins, to_process

    def mtd_structures(self):

        process_file = os.path.join(self.tmp_dir, "mtd_structures.json")

        progress, self.protein_list = self.check_process(process_file)

        for p in tqdm(self.protein_list, total=len(self.protein_list)):
            mtd_out_path = os.path.join(self.mtd_dir, f"{p}.json")
            struct_out_path = os.path.join(self.structures_dir, f"{p}.pdb")

            features = fetch_protein_data_uniprot(p)
            if features:
                toKeep = ['entryType', 'primaryAccession', 'secondaryAccessions', 'uniProtkbId', 'entryAudit', 'annotationScore', 'organism', 'proteinExistence', 'proteinDescription', 'genes', 'features', 'sequence']            
                mtd = {k: v for k, v in features.items() if k in toKeep}

                struct_res = fetch_protein_strcture_af(p)

                if struct_res:
                    pdb_url = struct_res[0].get('pdbUrl')
                    if pdb_url:
                        response = requests.get(pdb_url)
                        
                        # Save the structure
                        with open(struct_out_path, 'wb') as file:
                            file.write(response.content)
                            file.close()

                        # Save the mtd
                        with open(mtd_out_path, 'w') as file:
                            json.dump(mtd, file, indent=4)
                            file.close()

            progress.append(p)

            with open(process_file, "w") as f:
                json.dump(progress, f, indent=4)
                f.close()

        # If end
        if len(self.protein_list) == 0:
            self.protein_list = progress

            
    def process_proteins(self):

        all_for_assign = np.loadtxt(os.path.join(Path(__file__).parent.parent.parent.parent.parent, "config/all_assign.txt"))
        
        for p in tqdm(self.protein_list):
            if os.path.isfile(os.path.join(self.structures_dir, f"{p}.pdb")):
                
                if not (
                    os.path.exists(os.path.join(self.mtd_dir, f"{p}.json"))
                    and os.path.exists(os.path.join(self.structures_dir, f"{p}.pdb"))
                    and os.path.exists(os.path.join(self.processed_data_dir, f"{p}_r_contacts.npy"))
                    and os.path.exists(os.path.join(self.processed_data_dir, f"{p}_k_contacts.npy"))
                    and os.path.exists(os.path.join(self.processed_data_dir, f"{p}_nodes.pt"))
                ):
                
                    structure = open(os.path.join(self.structures_dir, f"{p}.pdb"))
                    r_contacts, k_contacts, x = self.pdb_to_cm(structure, 10)
                    x = self.match_feature(x, all_for_assign)

                    np.save(os.path.join(self.processed_data_dir, f"{p}_r_contacts.npy"), r_contacts)
                    np.save(os.path.join(self.processed_data_dir, f"{p}_k_contacts.npy"), k_contacts)
                    torch.save(x, os.path.join(self.processed_data_dir, f"{p}_nodes.pt"))

    def run(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.mtd_dir = os.path.join(self.tmp_dir, "mtd")
        os.makedirs(self.mtd_dir, exist_ok=True)

        self.structures_dir = os.path.join(self.tmp_dir, "structures")
        os.makedirs(self.structures_dir, exist_ok=True)

        logger.info("Protein structure and metadata retrieval...")

        self.mtd_structures()

        self.processed_data_dir = os.path.join(self.tmp_dir, "processed_data")
        os.makedirs(self.processed_data_dir, exist_ok=True)

        logger.info("Processing protein structures...")

        self.process_proteins()

        logger.info("Generating embeddings...")

        self.embedding_dir = os.path.join(self.tmp_dir, "embeddings")
        os.makedirs(self.embedding_dir, exist_ok=True)

        # Reniew - list
        self.protein_list = [p for p in self.protein_list if os.path.isfile(os.path.join(self.structures_dir,f'{p}.pdb'))]

        emb = Embedder(
            protein_list=self.protein_list,
            processed_data_dir=self.processed_data_dir,
            outdir=self.embedding_dir
        )

        emb.run()
        
def get_protein_list(data_dir):
        """
        Extracts a unique list of protein identifiers from the train, validation, and test datasets.

        Returns:
            list: A list of unique protein identifiers.
        """
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        dataset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        protein_list = dataset_df["P1"].unique().tolist()
        protein_list.extend(dataset_df["P2"].unique().tolist())

        protein_list = list(set(protein_list))

        return protein_list

if __name__ == "__main__":
    data_dir="/home/c3biolab/c3biolab_projects/doctorals/DPK/struct_ppi_pred/data"
    protein_list = get_protein_list(data_dir)

    protein_data_handler = ProteinDataHandler(protein_list=protein_list, session="DATASET")
    protein_data_handler.run()