import os

import torch
from torch_geometric.data import Dataset
import pandas as pd

from rdkit import Chem
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb

from featurize import Featurizer

class GraphDataset(Dataset): 
    def __init__(self, complex_ids, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.complex_ids = complex_ids 

        self.pdb_parser = PDBParser(QUIET=True, PERMISSIVE=1)
        self.featurizer = Featurizer() 
    
    def load_complex(self, complex_id): 
        pdb_path = f'{complex_id}_pocket.pdb'
        sdf_path = f'{complex_id}_ligand.sdf'

        try: 

            ligand = Chem.SDMolSupplier(sdf_path)
            protein = self.pdb_parser.get_structure(complex_id, pdb_path)

        except FileNotFoundError: 
            print(f'ERROR: Pocket file {complex_id} not found.')
        except Exception as e: 
            print(f'ERROR: Unexpected error on {complex_id}.')
            
            

    def process(self):
        for id in self.complex_ids: 

            protein, ligand = self.load_complex(id)

            data = self.featurizer.transform(protein, ligand)

            torch.save(data, f'{self.processed_dir}/{id}.pt')
                     



