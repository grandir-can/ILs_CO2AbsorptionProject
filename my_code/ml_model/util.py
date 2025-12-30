from rdkit import Chem
import torch
from torch_geometric.data import Data
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def process_desc(array):
    '''
    process descriptor, delete "NaN" in the descriptor and
    the dimensionality that is same in data inputs.
    '''

    desc_len = array.shape[1]
    rig_idx = []
    for i in range(desc_len):
        try:
            desc_range = array[:, i].max() - array[:, i].min()
            if desc_range != 0 and not np.isnan(desc_range):
                rig_idx.append(i)
        except:
            continue
    array = array[:, rig_idx]
    array = np.array(array, dtype=np.float32)
    return array

def rdkit_descripts(smiles):
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descriptors = [list(desc_calc.CalcDescriptors(mol)) for mol in mols]
    descriptors = np.array(descriptors)
    descriptors= process_desc(descriptors).tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(descriptors)
    pca = PCA(n_components=50)
    descriptors = pca.fit_transform(X_scaled)
    descriptors_dict = dict(zip(smiles, descriptors))
    return descriptors_dict

def morg_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    morg_fp = [Chem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    morg_fp_data = []
    for fp in morg_fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        morg_fp_data.append(arr)
    
    # Scale and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(morg_fp_data)
    pca = PCA(n_components=50)
    descriptors = pca.fit_transform(X_scaled)
    morg_dict = dict(zip(smiles, descriptors))
    return morg_dict


