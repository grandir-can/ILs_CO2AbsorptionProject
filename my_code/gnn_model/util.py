from rdkit import Chem
import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem as Chem

def smiles_to_data(smiles,tems,pres,props,type,target):
    if target=='CO2_capacity':
        symbols = [
            'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Fe',
        ]
    if target == 'viscosity':
        symbols = [
            'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Fe',
            'Al','Si'
        ]

    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    data_list = []
    i = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(smi)
            continue
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(symbols)
            symbol[symbols.index(atom.GetSymbol())] = 1.
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(hybridizations)
            hybridization[hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = atom.GetTotalNumHs()
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            if type == 'single':
                x = torch.tensor(symbol)

            elif type == 'mulriple':
                x = torch.tensor(symbol + [degree] + [formal_charge] +
                                 [radical_electrons] +
                                 [aromaticity] + [hydrogens] + [chirality] +
                                 chirality_type)

            xs.append(x)
        x = torch.stack(xs, dim=0)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.

            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring])

            edge_attrs += [edge_attr, edge_attr]
        if len(edge_attrs) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.stack(edge_attrs, dim=0)
        if props is None:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        t=tems[i], p=pres[i], smiles=smi)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        t=tems[i], p=pres[i], y=props[i], smiles=smi)

        data_list.append(data)
        i = i + 1
    return data_list