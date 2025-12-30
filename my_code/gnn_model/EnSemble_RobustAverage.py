from torch_geometric.data import Data, DataLoader
import pandas as pd
from joblib import load
import os
import numpy as np
from util import smiles_to_data
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error
import math
import argparse
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

def _normalize_target(target: str) -> str:
    key = target.strip().lower()
    if key in ('co2_capacity', 'co2', 'co2absorption', 'co2_absorption'):
        return 'CO2_capacity'
    if key in ('viscosity', 'vis'):
        return 'viscosity'
    raise ValueError(
        f"Unsupported target property: {target}. Use 'CO2_capacity' or 'viscosity'."
    )


def run_ensemble(target: str = 'CO2_capacity'):
    normalized_target = _normalize_target(target)

    if normalized_target == 'CO2_capacity':
        data_path = '../../data/CO2_capacity.xlsx'
        inds_path = '../../data/indexs/CO2_capacity'
        prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
    else:
        data_path = '../../data/viscosity.xlsx'
        inds_path = '../../data/indexs/viscosity'
        prop_col = 'log10(Viscosity,mPa.s)'

    df = pd.read_excel(data_path)
    smiles = df['smiles'].values
    if 'cation smiles' in df.columns:
        cat_smiles = df['cation smiles'].values
    else:
        cat_smiles = None
    if 'anion smiles' in df.columns:
        ani_smiles = df['anion smiles'].values
    else:
        ani_smiles = None
    props = df[prop_col].values

    tems = df['Temperature, K'].values
    pres = df['Pressure, kPa'].values

    transfer1 = MinMaxScaler(feature_range=(0, 1))
    transfer2 = MinMaxScaler(feature_range=(0, 1))
    tems = transfer1.fit_transform(tems.reshape(-1, 1))
    pres = transfer2.fit_transform(pres.reshape(-1, 1))

    all_data = smiles_to_data(smiles, tems, pres, props, 'single', normalized_target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R2Score_trains, R2Score_tests = [], []
    MSE_trains, MSE_tests = [], []
    RMSE_trains, RMSE_tests = [], []
    MAE_trains, MAE_tests = [], []

    model_base = os.path.join('../../model', normalized_target)

    for n in range(5):
        train_index = pd.read_csv(os.path.join(inds_path, r'train_ind_' + str(n) + '.csv'))
        test_index = pd.read_csv(os.path.join(inds_path, r'test_ind_' + str(n) + '.csv'))
        train_index = np.array(train_index['train_ind'])
        test_index = np.array(test_index['test_ind'])

        smiles_test, tems_test, pres_test = smiles[test_index], tems[test_index], pres[test_index]

        train_data, test_data = [], []
        for idx in train_index:
            train_data.append(all_data[idx])
        for idx in test_index:
            test_data.append(all_data[idx])

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        model_GIN = load(os.path.join(model_base, 'GIN', 'model_' + str(n) + '.pt')).to(device)
        model_GAT = load(os.path.join(model_base, 'GAT', 'model_' + str(n) + '.pt')).to(device)
        model_PNA = load(os.path.join(model_base, 'PNA', 'model_' + str(n) + '.pt')).to(device)
        model_MPNN = load(os.path.join(model_base, 'MPNN', 'model_' + str(n) + '.pt')).to(device)
        model_Attentive_FP = load(os.path.join(model_base, 'Attentive_FP', 'model_' + str(n) + '.pt')).to(device)

        smiles_train, tems_train, pres_train, y_train_preds, y_train_reals = [], [], [], [], []
        for data in train_loader:
            data = data.to(device)
            y_train_pred_GIN = model_GIN(data).cpu().detach().numpy()
            y_train_pred_PNA = model_PNA(data).cpu().detach().numpy()
            y_train_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1)
            y_train_pred_MPNN = model_MPNN(data).cpu().detach().numpy()
            y_train_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy()
            y_train_pred = np.array(
                [y_train_pred_PNA, y_train_pred_GAT, y_train_pred_MPNN, y_train_pred_Attentive_FP, y_train_pred_GIN]
            )
            y_train_real = data.y.cpu().numpy().tolist()

            y_train_pred_mean = np.mean(y_train_pred, axis=0)
            y_train_pred_std = np.std(y_train_pred, axis=0)

            y_train_pred_GIN_new, y_train_pred_PNA_new, y_train_pred_GAT_new, y_train_pred_MPNN_new, y_train_pred_Attentive_FP_new = [], [], [], [], []
            for i in range(len(y_train_pred_mean)):
                min = y_train_pred_mean[i] - y_train_pred_std[i]
                max = y_train_pred_mean[i] + y_train_pred_std[i]
                if min <= y_train_pred_GIN[i] <= max:
                    y_train_pred_GIN_new.append(y_train_pred_GIN[i])
                else:
                    y_train_pred_GIN_new.append(np.nan)

                if min <= y_train_pred_PNA[i] <= max:
                    y_train_pred_PNA_new.append(y_train_pred_PNA[i])
                else:
                    y_train_pred_PNA_new.append(np.nan)

                if min <= y_train_pred_GAT[i] <= max:
                    y_train_pred_GAT_new.append(y_train_pred_GAT[i])
                else:
                    y_train_pred_GAT_new.append(np.nan)

                if min <= y_train_pred_MPNN[i] <= max:
                    y_train_pred_MPNN_new.append(y_train_pred_MPNN[i])
                else:
                    y_train_pred_MPNN_new.append(np.nan)

                if min <= y_train_pred_Attentive_FP[i] <= max:
                    y_train_pred_Attentive_FP_new.append(y_train_pred_Attentive_FP[i])
                else:
                    y_train_pred_Attentive_FP_new.append(np.nan)
            y_train_pred_new = np.array(
                [y_train_pred_PNA_new, y_train_pred_GAT_new, y_train_pred_MPNN_new, y_train_pred_Attentive_FP_new,
                 y_train_pred_GIN_new])
            y_train_pred_mean_new = np.nanmean(y_train_pred_new, axis=0)

            t = transfer1.inverse_transform(data.t).squeeze()
            p = transfer2.inverse_transform(data.p).squeeze()

            smiles_train.extend(data.smiles)
            tems_train.extend(t)
            pres_train.extend(p)
            y_train_preds.extend(y_train_pred_mean_new)
            y_train_reals.extend(y_train_real)

        y_test_preds, y_test_reals = [], []
        for data in test_loader:
            data = data.to(device)
            y_test_pred_GIN = model_GIN(data).cpu().detach().numpy()
            y_test_pred_PNA = model_PNA(data).cpu().detach().numpy()
            y_test_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1)
            y_test_pred_MPNN = model_MPNN(data).cpu().detach().numpy()
            y_test_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy()
            y_test_pred = np.array(
                [y_test_pred_PNA , y_test_pred_GAT, y_test_pred_MPNN,y_test_pred_Attentive_FP,y_test_pred_GIN]
            )

            y_test_pred_mean = np.mean(y_test_pred,axis =0)
            y_test_pred_std = np.std(y_test_pred, axis=0)

            y_test_pred_GIN_new, y_test_pred_PNA_new, y_test_pred_GAT_new, y_test_pred_MPNN_new, y_test_pred_Attentive_FP_new = [], [], [], [], []
            for i in range(len(y_test_pred_mean)):
                min = y_test_pred_mean[i] - y_test_pred_std[i]
                max = y_test_pred_mean[i] + y_test_pred_std[i]
                if min <= y_test_pred_GIN[i] <= max:
                    y_test_pred_GIN_new.append(y_test_pred_GIN[i])
                else:
                    y_test_pred_GIN_new.append(np.nan)

                if min <= y_test_pred_PNA[i] <= max:
                    y_test_pred_PNA_new.append(y_test_pred_PNA[i])
                else:
                    y_test_pred_PNA_new.append(np.nan)

                if min <= y_test_pred_GAT[i] <= max:
                    y_test_pred_GAT_new.append(y_test_pred_GAT[i])
                else:
                    y_test_pred_GAT_new.append(np.nan)

                if min <= y_test_pred_MPNN[i] <= max:
                    y_test_pred_MPNN_new.append(y_test_pred_MPNN[i])
                else:
                    y_test_pred_MPNN_new.append(np.nan)

                if min <= y_test_pred_Attentive_FP[i] <= max:
                    y_test_pred_Attentive_FP_new.append(y_test_pred_Attentive_FP[i])
                else:
                    y_test_pred_Attentive_FP_new.append(np.nan)
            y_test_pred_new = np.array(
                [y_test_pred_PNA_new, y_test_pred_GAT_new, y_test_pred_MPNN_new, y_test_pred_Attentive_FP_new,
                 y_test_pred_GIN_new])

            y_test_pred_mean_new = np.nanmean(y_test_pred_new, axis=0)


            y_test_real = data.y.cpu().numpy().tolist()
            y_test_preds.extend(y_test_pred_mean_new)
            y_test_reals.extend(y_test_real)

        R2Score_train = r2_score(y_train_reals,y_train_preds)
        R2Score_trains.append(R2Score_train)
        R2Score_test = r2_score(y_test_reals, y_test_preds)
        R2Score_tests.append(R2Score_test)


        MSE_train = mean_squared_error(y_train_reals,y_train_preds)
        MSE_trains.append(MSE_train)
        MSE_test = mean_squared_error(y_test_reals, y_test_preds)
        MSE_tests.append(MSE_test)

        RMSE_train = math.sqrt(MSE_train)
        RMSE_trains.append(RMSE_train)
        RMSE_test = math.sqrt(MSE_test)
        RMSE_tests.append(RMSE_test)

        MAE_train = mean_absolute_error(y_train_reals,y_train_preds)
        MAE_trains.append(MAE_train)
        MAE_test = mean_absolute_error(y_test_reals, y_test_preds)
        MAE_tests.append(MAE_test)

    train_Score = [str(round(np.mean(R2Score_trains), 4)) + '+-' + str(round(np.std(R2Score_trains), 4)),
                   str(round(np.mean(MSE_trains), 4)) + '+-' + str(round(np.std(MSE_trains), 4)),
                   str(round(np.mean(RMSE_trains), 4)) + '+-' + str(round(np.std(RMSE_trains), 4)),
                   str(round(np.mean(MAE_trains), 4)) + '+-' + str(round(np.std(MAE_trains), 4))]
    test_Score = [str(round(np.mean(R2Score_tests), 4)) + '+-' + str(round(np.std(R2Score_tests), 4)),
                  str(round(np.mean(MSE_tests), 4)) + '+-' + str(round(np.std(MSE_tests), 4)),
                  str(round(np.mean(RMSE_tests), 4)) + '+-' + str(round(np.std(RMSE_tests), 4)),
                  str(round(np.mean(MAE_tests), 4)) + '+-' + str(round(np.std(MAE_tests), 4))]

    df_Score = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score},
                               index=['R2', 'MSE', 'RMSE', 'MAE'])

    print(R2Score_tests)
    print(df_Score)
    # df_Score.to_csv('../../data/result/viscosity/Ensemble/mean_std_score.csv')

    df_Score = pd.DataFrame({'Train R2': R2Score_trains, 'train MSE': MSE_trains,'train RMSE': RMSE_trains,'train MAE': MAE_trains,
                             'Test R2': R2Score_tests, 'test MSE': MSE_tests,'test RMSE': RMSE_tests,'test MAE': MAE_tests},)
    print(df_Score)
    # df_Score.to_csv('../../data/result/viscosity/Ensemble/mean_std_score_5_fold.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble robust average evaluation for GNN models on IL datasets.'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='CO2_capacity',
        help="Target property: 'CO2_capacity' (CO2 absorption) or 'viscosity'. You can also use aliases like 'co2' or 'vis'."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_ensemble(args.target)