import pandas as pd
import numpy as np
import pickle
from util import rdkit_descripts, morg_fingerprints

def generate_and_save_features():
    """
    Loads CO2 and viscosity data, generates RDKit descriptors and Morgan fingerprints
    for all unique SMILES strings, and saves them to pickle files.
    """
    # Load datasets
    co2_data_path = '../../data/CO2_capacity.xlsx'
    viscosity_data_path = '../../data/viscosity.xlsx'

    df_co2 = pd.read_excel(co2_data_path)
    df_viscosity = pd.read_excel(viscosity_data_path)

    # Get unique SMILES from both datasets
    all_smiles = pd.concat([df_co2['smiles'], df_viscosity['smiles']]).unique()
    print(f'Found {len(all_smiles)} unique SMILES strings.')

    # Generate and save RDKit descriptors
    print('Generating RDKit descriptors...')
    rdkit_features = rdkit_descripts(all_smiles)
    with open('../../data/rdkit_features.pkl', 'wb') as f:
        pickle.dump(rdkit_features, f)
    print('RDKit descriptors saved to ../../data/rdkit_features.pkl')

    # Generate and save Morgan fingerprints
    print('Generating Morgan fingerprints...')
    morgan_features = morg_fingerprints(all_smiles)
    with open('../../data/morgan_features.pkl', 'wb') as f:
        pickle.dump(morgan_features, f)
    print('Morgan fingerprints saved to ../../morgan_features.pkl')

if __name__ == '__main__':
    generate_and_save_features()
