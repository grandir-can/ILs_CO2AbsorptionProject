from torch_geometric.data import DataLoader
import pandas as pd
from joblib import load
import os
import numpy as np
from util import smiles_to_data
from sklearn.preprocessing import MinMaxScaler

import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import joblib
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


def _normalize_meta(meta: str) -> str:
    key = meta.strip().lower()
    if key in ('extra_trees', 'extratrees', 'extra', 'et', 'extratreesregressor'):
        return 'extra_trees'
    if key in ('linear', 'lr', 'linearregression'):
        return 'linear'
    raise ValueError(
        f"Unsupported meta-learner: {meta}. Use 'extra_trees' or 'linear'."
    )


def run_stacking_ensemble(target: str = 'viscosity', meta: str = 'extra_trees'):
    normalized_target = _normalize_target(target)
    normalized_meta = _normalize_meta(meta)

    if normalized_target == 'CO2_capacity':
        data_path = os.path.join('..', '..', 'data', 'CO2_capacity.xlsx')
        inds_path = os.path.join('..', '..', 'data', 'indexs', 'CO2_capacity')
        prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
    else:
        data_path = os.path.join('..', '..', 'data', 'viscosity.xlsx')
        inds_path = os.path.join('..', '..', 'data', 'indexs', 'viscosity')
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

    # 与 EnSemble_RobustAverage/EnSemble_Weighted 保持一致
    all_data = smiles_to_data(smiles, tems, pres, props, 'single', normalized_target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    R2Score_trains, R2Score_tests = [], []
    MSE_trains, MSE_tests = [], []
    RMSE_trains, RMSE_tests = [], []
    MAE_trains, MAE_tests = [], []

    model_base = os.path.join('..', '..', 'model', normalized_target)

    result_base = os.path.join('..', '..', 'data', 'result', normalized_target, 'Ensemble_Stacking')
    train_dir = os.path.join(result_base, 'train_2')
    test_dir = os.path.join(result_base, 'test_2')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for n in range(5):
        train_index = pd.read_csv(os.path.join(inds_path, f'train_ind_{n}.csv'))
        test_index = pd.read_csv(os.path.join(inds_path, f'test_ind_{n}.csv'))
        train_index = np.array(train_index['train_ind'])
        test_index = np.array(test_index['test_ind'])

        train_data = [all_data[idx] for idx in train_index]
        test_data = [all_data[idx] for idx in test_index]

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        # 加载 5 个基学习器
        model_GIN = load(os.path.join(model_base, 'GIN', f'model_{n}.pt')).to(device)
        model_GAT = load(os.path.join(model_base, 'GAT', f'model_{n}.pt')).to(device)
        pna_name = f'model_{n}.pt'
        model_PNA = load(os.path.join(model_base, 'PNA', pna_name)).to(device)
        model_MPNN = load(os.path.join(model_base, 'MPNN', f'model_{n}.pt')).to(device)
        model_Attentive_FP = load(os.path.join(model_base, 'Attentive_FP', f'model_{n}.pt')).to(device)

        # 训练集：生成基学习器预测作为元学习器输入特征
        smiles_train, tems_train, pres_train = [], [], []
        y_train_real_all = []
        y_train_pred_GIN_all, y_train_pred_PNA_all, y_train_pred_GAT_all, y_train_pred_MPNN_all, y_train_pred_Attentive_FP_all = [], [], [], [], []
        for data in train_loader:
            data = data.to(device)
            y_train_pred_GIN = model_GIN(data).cpu().detach().numpy().reshape(-1).tolist()
            y_train_pred_PNA = model_PNA(data).cpu().detach().numpy().reshape(-1).tolist()
            y_train_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1).tolist()
            y_train_pred_MPNN = model_MPNN(data).cpu().detach().numpy().reshape(-1).tolist()
            y_train_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy().reshape(-1).tolist()

            y_train_pred_GIN_all.extend(y_train_pred_GIN)
            y_train_pred_PNA_all.extend(y_train_pred_PNA)
            y_train_pred_GAT_all.extend(y_train_pred_GAT)
            y_train_pred_MPNN_all.extend(y_train_pred_MPNN)
            y_train_pred_Attentive_FP_all.extend(y_train_pred_Attentive_FP)

            y_train_real = data.y.cpu().numpy().tolist()
            y_train_real_all.extend(y_train_real)

            t = transfer1.inverse_transform(data.t).squeeze()
            p = transfer2.inverse_transform(data.p).squeeze()
            smiles_train.extend(data.smiles)
            tems_train.extend(t)
            pres_train.extend(p)

        # 特征顺序统一为 [PNA, GAT, MPNN, Attentive_FP, GIN]
        stacked_X_train = np.column_stack((
            np.array(y_train_pred_PNA_all),
            np.array(y_train_pred_GAT_all),
            np.array(y_train_pred_MPNN_all),
            np.array(y_train_pred_Attentive_FP_all),
            np.array(y_train_pred_GIN_all),
        ))

        # 选择元学习器
        if normalized_meta == 'extra_trees':
            # ExtraTreesRegressor + 网格搜索
            param_grid = {
                'n_estimators': [100, 200, 400, 600, 800],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [1, 2, 3, 4],
                'min_samples_leaf': [1, 2, 3, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
            }
            base_meta = ExtraTreesRegressor(random_state=42)
            grid_search = GridSearchCV(estimator=base_meta, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(stacked_X_train, y_train_real_all)
            print(f"Fold {n} best ExtraTrees params:", grid_search.best_params_)
            print("Best CV score:", grid_search.best_score_)
            best_model = grid_search.best_estimator_
        else:

            param_grid = {
                'copy_X':[True,False],
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
            base_meta = LinearRegression(random_state=42)
            grid_search = GridSearchCV(estimator=base_meta, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(stacked_X_train, y_train_real_all)
            print(f"Fold {n} best ExtraTrees params:", grid_search.best_params_)
            print("Best CV score:", grid_search.best_score_)
            best_model = grid_search.best_estimator_

        y_train_ensemble = best_model.predict(stacked_X_train)
        if normalized_target == 'CO2_capacity':
            y_train_ensemble = np.where(y_train_ensemble < 0, 1e-8, y_train_ensemble)

        # 测试集：生成基学习器预测并通过元学习器得到集成预测
        y_test_real_all = []
        smiles_test, tems_test, pres_test = [], [], []
        y_test_pred_GIN_all, y_test_pred_PNA_all, y_test_pred_GAT_all, y_test_pred_MPNN_all, y_test_pred_Attentive_FP_all = [], [], [], [], []
        for data in test_loader:
            data = data.to(device)
            y_test_pred_GIN = model_GIN(data).cpu().detach().numpy().reshape(-1).tolist()
            y_test_pred_PNA = model_PNA(data).cpu().detach().numpy().reshape(-1).tolist()
            y_test_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1).tolist()
            y_test_pred_MPNN = model_MPNN(data).cpu().detach().numpy().reshape(-1).tolist()
            y_test_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy().reshape(-1).tolist()

            y_test_pred_GIN_all.extend(y_test_pred_GIN)
            y_test_pred_PNA_all.extend(y_test_pred_PNA)
            y_test_pred_GAT_all.extend(y_test_pred_GAT)
            y_test_pred_MPNN_all.extend(y_test_pred_MPNN)
            y_test_pred_Attentive_FP_all.extend(y_test_pred_Attentive_FP)

            y_test_real = data.y.cpu().numpy().tolist()
            y_test_real_all.extend(y_test_real)

            t = transfer1.inverse_transform(data.t).squeeze()
            p = transfer2.inverse_transform(data.p).squeeze()
            smiles_test.extend(data.smiles)
            tems_test.extend(t)
            pres_test.extend(p)

        stacked_X_test = np.column_stack((
            np.array(y_test_pred_PNA_all),
            np.array(y_test_pred_GAT_all),
            np.array(y_test_pred_MPNN_all),
            np.array(y_test_pred_Attentive_FP_all),
            np.array(y_test_pred_GIN_all),
        ))

        y_test_ensemble = best_model.predict(stacked_X_test)
        if normalized_target == 'CO2_capacity':
            y_test_ensemble = np.where(y_test_ensemble < 0, 1e-8, y_test_ensemble)

        # 本折指标
        R2Score_train = r2_score(y_train_real_all, y_train_ensemble)
        R2Score_trains.append(R2Score_train)
        R2Score_test = r2_score(y_test_real_all, y_test_ensemble)
        R2Score_tests.append(R2Score_test)

        MSE_train = mean_squared_error(y_train_real_all, y_train_ensemble)
        MSE_trains.append(MSE_train)
        MSE_test = mean_squared_error(y_test_real_all, y_test_ensemble)
        MSE_tests.append(MSE_test)

        RMSE_train = math.sqrt(MSE_train)
        RMSE_trains.append(RMSE_train)
        RMSE_test = math.sqrt(MSE_test)
        RMSE_tests.append(RMSE_test)

        MAE_train = mean_absolute_error(y_train_real_all, y_train_ensemble)
        MAE_trains.append(MAE_train)
        MAE_test = mean_absolute_error(y_test_real_all, y_test_ensemble)
        MAE_tests.append(MAE_test)

        dataframe = pd.DataFrame({
            'smiles': smiles_train,
            'Temperature, K': tems_train,
            'Pressure, kPa': pres_train,
            'y_train': y_train_real_all,
            'y_train_pred': y_train_ensemble,
        })
        # dataframe.to_csv(os.path.join(train_dir, f'real_pred_train_{n}.csv'), sep=',', index=False)

        dataframe = pd.DataFrame({
            'smiles': smiles_test,
            'Temperature, K': tems_test,
            'Pressure, kPa': pres_test,
            'y_test': y_test_real_all,
            'y_test_pred': y_test_ensemble,
        })
        # dataframe.to_csv(os.path.join(test_dir, f'real_pred_test_{n}.csv'), sep=',', index=False)

    train_Score = [
        str(round(np.mean(R2Score_trains), 4)) + '+-' + str(round(np.std(R2Score_trains), 4)),
        str(round(np.mean(MSE_trains), 4)) + '+-' + str(round(np.std(MSE_trains), 4)),
        str(round(np.mean(RMSE_trains), 4)) + '+-' + str(round(np.std(RMSE_trains), 4)),
        str(round(np.mean(MAE_trains), 4)) + '+-' + str(round(np.std(MAE_trains), 4)),
    ]
    test_Score = [
        str(round(np.mean(R2Score_tests), 4)) + '+-' + str(round(np.std(R2Score_tests), 4)),
        str(round(np.mean(MSE_tests), 4)) + '+-' + str(round(np.std(MSE_tests), 4)),
        str(round(np.mean(RMSE_tests), 4)) + '+-' + str(round(np.std(RMSE_tests), 4)),
        str(round(np.mean(MAE_tests), 4)) + '+-' + str(round(np.std(MAE_tests), 4)),
    ]

    df_Score_summary = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score}, index=['R2', 'MSE', 'RMSE', 'MAE'])
    print('Fold-wise test R2:', R2Score_tests)
    print('Summary scores:')
    print(df_Score_summary)

    summary_path = os.path.join(result_base, 'ensemble_Stacking2.csv')
    df_Score_summary.to_csv(summary_path)

    df_Score_all = pd.DataFrame({
        'Train R2': R2Score_trains,
        'train MSE': MSE_trains,
        'train RMSE': RMSE_trains,
        'train MAE': MAE_trains,
        'Test R2': R2Score_tests,
        'test MSE': MSE_tests,
        'test RMSE': RMSE_tests,
        'test MAE': MAE_tests,
    })
    print('Scores for each fold:')
    print(df_Score_all)

    scores_path = os.path.join(result_base, 'ensemble_Stacking_score_5_fold2.csv')
    # df_Score_all.to_csv(scores_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stacking ensemble evaluation for GNN models on IL datasets.'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='viscosity',
        help=(
            "Target property: 'CO2_capacity' (CO2 absorption) or 'viscosity'. "
            "You can also use aliases like 'co2' or 'vis'."
        ),
    )
    parser.add_argument(
        '--meta',
        type=str,
        default='extra_trees',
        help=(
            "Meta learner: 'extra_trees' (ExtraTreesRegressor) or 'linear' (LinearRegression). "
            "You can also use aliases like 'et' or 'lr'."
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_stacking_ensemble(target=args.target, meta=args.meta)
