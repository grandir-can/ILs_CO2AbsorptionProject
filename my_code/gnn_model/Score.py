from torch_geometric.data import Data, DataLoader
import pandas as pd
from joblib import load
import os
import numpy as np
from util import smiles_to_data
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error
import math
import sys
import types
from gnn import GAT as _GAT_cls, GIN as _GIN_cls, PNA as _PNA_cls, MPNN as _MPNN_cls, AttentiveFP as _AttentiveFP_cls, GATEConv as _GATEConv_cls

def _register_legacy_module(module_name: str, cls_name: str, cls_obj):
    """Register a lightweight legacy module in sys.modules for unpickling.

    Old models were pickled with class paths like "MPNN.MPNN" or "GAT.GAT".
    We create in-memory modules with matching names so joblib.load can find
    the classes, without requiring separate .py files on disk.
    """

    # Reuse existing shim module if present so we can attach multiple
    # classes (e.g., AttentiveFP and its nested GATEConv) to the same module.
    mod = sys.modules.get(module_name)
    if mod is None:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod

    setattr(mod, cls_name, cls_obj)


# Register legacy modules for all five GNN architectures.
_register_legacy_module('MPNN', 'MPNN', _MPNN_cls)
_register_legacy_module('GAT', 'GAT', _GAT_cls)
_register_legacy_module('GIN', 'GIN', _GIN_cls)
_register_legacy_module('PNA', 'PNA', _PNA_cls)
_register_legacy_module('AttentiveFP', 'AttentiveFP', _AttentiveFP_cls)
_register_legacy_module('AttentiveFP', 'GATEConv', _GATEConv_cls)

parser = argparse.ArgumentParser(description='Evaluate trained GNN models on IL datasets.')
parser.add_argument('--target', type=str, default='CO2_capacity',
                    help='Target property: "CO2_capacity" or "viscosity" (case-insensitive).')
parser.add_argument('--model', type=str, default='MPNN',
                    help='GNN model name, e.g. MPNN, GAT, GIN, PNA, AttentiveFP.')
parser.add_argument('--folds', type=int, default=5,
                    help='Number of folds to evaluate (default: 5).')
parser.add_argument('--data-dir', type=str, default='../../data',
                    help='Base data directory (default: ../../data).')
parser.add_argument('--index-base', type=str, default='../../data/indexs',
                    help='Base index directory containing CO2_capacity/ and viscosity/ (default: ../../data/indexs).')
parser.add_argument('--model-base', type=str, default='../../model',
                    help='Base directory where trained models are saved (default: ../../model).')

args = parser.parse_args()

target_key = args.target.strip().lower()
if target_key in ['co2_capacity', 'co2', 'co2capacity']:
    normalized_target = 'CO2_capacity'
    data_path = os.path.join(args.data_dir, 'CO2_capacity.xlsx')
    inds_path = os.path.join(args.index_base, 'CO2_capacity')
    prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
elif target_key in ['viscosity', 'vis']:
    normalized_target = 'viscosity'
    data_path = os.path.join(args.data_dir, 'viscosity.xlsx')
    inds_path = os.path.join(args.index_base, 'viscosity')
    prop_col = 'log10(Viscosity,mPa.s)'
else:
    raise ValueError('Unsupported target property: {}. Use "CO2_capacity" or "viscosity".'.format(args.target))

model_key = args.model.strip().lower()
if model_key in ['attentivefp', 'attentive_fp', 'attentive']:
    model_folder = 'Attentive_FP'
else:
    model_folder = model_key.upper()
model_dir = os.path.join(args.model_base, normalized_target, model_folder)

df = pd.read_excel(data_path)
smiles = df['smiles'].values
cat_smiles = df['cation smiles'].values
ani_smiles = df['anion smiles'].values
props = df[prop_col].values #log10(Viscosity,mPa.s)  Mole_fraction_of_carbon_dioxide[Liquid] Heat Capacity,J/K/mol Density,kg/m3

tems = df['Temperature, K'].values
pres = df['Pressure, kPa'].values

transfer1 = MinMaxScaler(feature_range=(0, 1))
transfer2 = MinMaxScaler(feature_range=(0, 1))
tems_scale = transfer1.fit_transform(tems.reshape(-1,1))
pres_scale = transfer2.fit_transform(pres.reshape(-1,1))

t = transfer1.inverse_transform(tems_scale).squeeze()
p = transfer2.inverse_transform(pres_scale).squeeze()
all_data = smiles_to_data(smiles, tems_scale, pres_scale,props,'single',args.target)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
R2Score_trains, R2Score_tests = [], []
MSE_trains, MSE_tests = [], []
RMSE_trains, RMSE_tests = [], []
MAE_trains, MAE_tests = [], []
for i in range(args.folds):
    train_index = pd.read_csv(os.path.join(inds_path, r'train_ind_' + str(i) + '.csv'))
    test_index = pd.read_csv(os.path.join(inds_path, r'test_ind_' + str(i) + '.csv'))
    train_index = np.array(train_index['train_ind'])
    test_index = np.array(test_index['test_ind'])

    smiles_test,tems_test,pres_test = smiles[test_index],tems[test_index],pres[test_index]

    train_data,test_data = [],[]
    for idx in train_index:
        train_data.append(all_data[idx])
    for idx in test_index:
        test_data.append(all_data[idx])

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model_path = os.path.join(model_dir, 'model_{}.pt'.format(i))
    model = load(model_path)
    model = model.to(device)
    # print(model)

    smiles_train,tems_train,pres_train,y_train_preds,y_train_reals = [],[],[],[],[]
    for data in train_loader:
        data = data.to(device)
        y_train_pred = model(data).squeeze().cpu().detach().numpy().tolist()
        y_train_real = data.y.cpu().numpy().tolist()
        t = transfer1.inverse_transform(data.t).squeeze()
        p = transfer2.inverse_transform(data.p).squeeze()

        smiles_train.extend(data.smiles)
        tems_train.extend(t)
        pres_train.extend(p)
        y_train_preds.extend(y_train_pred)
        y_train_reals.extend(y_train_real)
    y_test_preds, y_test_reals = [], []
    for data in test_loader:
        data = data.to(device)
        y_test_pred = model(data).squeeze().cpu().detach().numpy().tolist()
        y_test_real = data.y.cpu().numpy().tolist()
        if isinstance(y_test_pred,list):
            y_test_preds.extend(y_test_pred)
        else:
            y_test_preds.append(y_test_pred)

        y_test_reals.extend(y_test_real)

    if normalized_target == 'CO2_capacity':
        _min_pred_value = 1e-8
        y_train_eval = [pred if pred >= 0 else _min_pred_value for pred in y_train_preds]
        y_test_eval = [pred if pred >= 0 else _min_pred_value for pred in y_test_preds]
    else:
        y_train_eval = y_train_preds
        y_test_eval = y_test_preds

    R2Score_train = r2_score(y_train_reals, y_train_eval)
    R2Score_trains.append(R2Score_train)
    R2Score_test = r2_score(y_test_reals, y_test_eval)
    R2Score_tests.append(R2Score_test)

    MSE_train = mean_squared_error(y_train_reals, y_train_eval)
    MSE_trains.append(MSE_train)
    MSE_test = mean_squared_error(y_test_reals, y_test_eval)
    MSE_tests.append(MSE_test)

    RMSE_train = math.sqrt(MSE_train)
    RMSE_trains.append(RMSE_train)
    RMSE_test = math.sqrt(MSE_test)
    RMSE_tests.append(RMSE_test)

    MAE_train = mean_absolute_error(y_train_reals, y_train_eval)
    MAE_trains.append(MAE_train)
    MAE_test = mean_absolute_error(y_test_reals, y_test_eval)
    MAE_tests.append(MAE_test)

train_Score = [str(round(np.mean(R2Score_trains), 4)) + '+-' + str(round(np.std(R2Score_trains), 4)),
               str(round(np.mean(MSE_trains), 4)) + '+-' + str(round(np.std(MSE_trains), 4)),
               str(round(np.mean(RMSE_trains), 4)) + '+-' + str(round(np.std(RMSE_trains), 4)),
               str(round(np.mean(MAE_trains), 4)) + '+-' + str(round(np.std(MAE_trains), 4))]
test_Score = [str(round(np.mean(R2Score_tests), 4)) + '+-' + str(round(np.std(R2Score_tests), 4)),
              str(round(np.mean(MSE_tests), 4)) + '+-' + str(round(np.std(MSE_tests), 4)),
              str(round(np.mean(RMSE_tests), 4)) + '+-' + str(round(np.std(RMSE_tests), 4)),
              str(round(np.mean(MAE_tests), 4)) + '+-' + str(round(np.std(MAE_tests), 4))]

df_Score = pd.DataFrame({'Train R2': R2Score_trains, 'train MSE': MSE_trains,'train RMSE': RMSE_trains,'train MAE': MAE_trains,
                         'Test R2': R2Score_tests, 'mean_test MSE': MSE_tests,'mean_test RMSE': RMSE_tests,'mean_test MAE': MAE_tests},)
print(test_Score)
print(R2Score_tests)
print(df_Score)
# df_Score.to_csv('../../data/result/CH4_capacity/Attentive_FP/data_point_split/score.csv')

mean_df_Score = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score},
                            index=['R2', 'MSE', 'RMSE', 'MAE'])
print(mean_df_Score)
# mean_df_Score.to_csv('../../data/result/CH4_capacity/Attentive_FP/data_point_split/score_mean.csv')








