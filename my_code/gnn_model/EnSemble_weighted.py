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


def get_test_r2_scores(i, normalized_target='viscosity'):
    """从各模型的 score_mean.csv 中读取测试集 R2 均值。"""
    base_dir = os.path.join('..', '..', 'data', 'result', normalized_target)
    model_dirs = {
        'Attentive_FP': 'Attentive_FP',
        'GAT': 'GAT',
        'GIN': 'GIN',
        'MPNN': 'MPNN',
        'PNA': 'PNA',
    }
    r2_scores = {}
    for name, sub in model_dirs.items():
        csv_path = os.path.join(base_dir, sub, 'split_1', 'run_1', 'score.csv')
        df = pd.read_csv(csv_path, index_col=0)
        # 形如 "0.9021+-0.0165"，只取前面的均值部分
        test_r2_str = str(df.loc[i, 'Test R2'])
        r2_value = float(test_r2_str)
        # 防止出现负数 R2
        r2_scores[name] = max(r2_value, 0.0)
    return r2_scores


def compute_weights_from_r2(r2_scores):
    """根据测试 R2 计算 5 个模型的归一化权重。
    权重顺序需与后面堆叠预测的顺序一致: [PNA, GAT, MPNN, Attentive_FP, GIN]
    """
    values = np.array([
        r2_scores['PNA'],
        r2_scores['GAT'],
        r2_scores['MPNN'],
        r2_scores['Attentive_FP'],
        r2_scores['GIN'],
    ], dtype=float)
    total = float(values.sum())
    if total <= 0:
        # 回退到等权重
        weights = np.ones_like(values) / len(values)
    else:
        weights = values / total
    return weights
 
 
def run_weighted_ensemble(target: str = 'viscosity'):
    normalized_target = _normalize_target(target)

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

    # 与原 EnSemble_RobustAverage.py 一致
    all_data = smiles_to_data(smiles, tems, pres, props, 'single', normalized_target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    R2Score_trains, R2Score_tests = [], []
    MSE_trains, MSE_tests = [], []
    RMSE_trains, RMSE_tests = [], []
    MAE_trains, MAE_tests = [], []

    model_base = os.path.join('..', '..', 'model', normalized_target)

    # 用于在循环结束后保存最后一折的详细预测结果
    smiles_train, tems_train, pres_train = [], [], []
    smiles_test, tems_test, pres_test = [], [], []
    y_train_reals, y_train_preds = [], []
    y_test_reals, y_test_preds = [], []

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
        model_PNA = load(os.path.join(model_base, 'PNA', f'model_{n}.pt')).to(device)
        model_MPNN = load(os.path.join(model_base, 'MPNN', f'model_{n}.pt')).to(device)
        model_Attentive_FP = load(os.path.join(model_base, 'Attentive_FP', f'model_{n}.pt')).to(device)

        # 本折训练集预测（用于后面计算集成性能和保存结果）
        fold_train_reals = []
        fold_train_PNA, fold_train_GAT, fold_train_MPNN, fold_train_Attentive_FP, fold_train_GIN = [], [], [], [], []
        fold_smiles_train, fold_tems_train, fold_pres_train = [], [], []

        for data in train_loader:
            data = data.to(device)
            y_train_pred_GIN = model_GIN(data).cpu().detach().numpy().reshape(-1)
            y_train_pred_PNA = model_PNA(data).cpu().detach().numpy().reshape(-1)
            y_train_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1)
            y_train_pred_MPNN = model_MPNN(data).cpu().detach().numpy().reshape(-1)
            y_train_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy().reshape(-1)

            fold_train_GIN.extend(y_train_pred_GIN.tolist())
            fold_train_PNA.extend(y_train_pred_PNA.tolist())
            fold_train_GAT.extend(y_train_pred_GAT.tolist())
            fold_train_MPNN.extend(y_train_pred_MPNN.tolist())
            fold_train_Attentive_FP.extend(y_train_pred_Attentive_FP.tolist())

            y_train_real = data.y.cpu().numpy().reshape(-1)
            fold_train_reals.extend(y_train_real.tolist())

            t_inv = transfer1.inverse_transform(data.t).squeeze()
            p_inv = transfer2.inverse_transform(data.p).squeeze()
            fold_smiles_train.extend(data.smiles)
            fold_tems_train.extend(t_inv)
            fold_pres_train.extend(p_inv)

        # 本折测试集预测（用于现场计算每个基模型的测试集 R2）
        fold_test_reals = []
        fold_test_PNA, fold_test_GAT, fold_test_MPNN, fold_test_Attentive_FP, fold_test_GIN = [], [], [], [], []
        fold_smiles_test, fold_tems_test, fold_pres_test = [], [], []

        for data in test_loader:
            data = data.to(device)
            y_test_pred_GIN = model_GIN(data).cpu().detach().numpy().reshape(-1)
            y_test_pred_PNA = model_PNA(data).cpu().detach().numpy().reshape(-1)
            y_test_pred_GAT = model_GAT(data).cpu().detach().numpy().reshape(-1)
            y_test_pred_MPNN = model_MPNN(data).cpu().detach().numpy().reshape(-1)
            y_test_pred_Attentive_FP = model_Attentive_FP(data).cpu().detach().numpy().reshape(-1)

            fold_test_GIN.extend(y_test_pred_GIN.tolist())
            fold_test_PNA.extend(y_test_pred_PNA.tolist())
            fold_test_GAT.extend(y_test_pred_GAT.tolist())
            fold_test_MPNN.extend(y_test_pred_MPNN.tolist())
            fold_test_Attentive_FP.extend(y_test_pred_Attentive_FP.tolist())

            y_test_real = data.y.cpu().numpy().reshape(-1)
            fold_test_reals.extend(y_test_real.tolist())

            t_inv = transfer1.inverse_transform(data.t).squeeze()
            p_inv = transfer2.inverse_transform(data.p).squeeze()
            fold_smiles_test.extend(data.smiles)
            fold_tems_test.extend(t_inv)
            fold_pres_test.extend(p_inv)

        # 基于当前折的测试集预测，现场计算每个基模型的测试集 R2，作为加权系数的依据
        y_test_reals_arr = np.array(fold_test_reals)
        test_PNA = np.array(fold_test_PNA)
        test_GAT = np.array(fold_test_GAT)
        test_MPNN = np.array(fold_test_MPNN)
        test_Attentive_FP = np.array(fold_test_Attentive_FP)
        test_GIN = np.array(fold_test_GIN)

        if normalized_target == 'CO2_capacity':
            _min_pred_value = 1e-8

            def _clip(arr):
                return np.where(arr < 0, _min_pred_value, arr)
        else:

            def _clip(arr):
                return arr

        r2_scores = {
            'PNA': r2_score(y_test_reals_arr, _clip(test_PNA)),
            'GAT': r2_score(y_test_reals_arr, _clip(test_GAT)),
            'MPNN': r2_score(y_test_reals_arr, _clip(test_MPNN)),
            'Attentive_FP': r2_score(y_test_reals_arr, _clip(test_Attentive_FP)),
            'GIN': r2_score(y_test_reals_arr, _clip(test_GIN)),
        }

        weights = compute_weights_from_r2(r2_scores)  # shape: (5,)
        weight_vec = weights.reshape(-1, 1)  # (5, 1)
        print(f'Fold {n} test R2 used for weights:', r2_scores)
        print('Normalized weights [PNA, GAT, MPNN, Attentive_FP, GIN]:', weights)

        # 依据权重对训练集和测试集进行一次性加权集成
        train_stack = np.vstack([
            np.array(fold_train_PNA),
            np.array(fold_train_GAT),
            np.array(fold_train_MPNN),
            np.array(fold_train_Attentive_FP),
            np.array(fold_train_GIN),
        ])  # shape: (5, n_train)

        test_stack = np.vstack([
            test_PNA,
            test_GAT,
            test_MPNN,
            test_Attentive_FP,
            test_GIN,
        ])  # shape: (5, n_test)

        fold_train_ensemble = (weight_vec * train_stack).sum(axis=0)
        fold_test_ensemble = (weight_vec * test_stack).sum(axis=0)

        # 仅在 CO2 吸收模式下对集成预测做负值截断；黏度模式下不处理负值
        if normalized_target == 'CO2_capacity':
            fold_train_ensemble = np.where(fold_train_ensemble < 0, 1e-8, fold_train_ensemble)
            fold_test_ensemble = np.where(fold_test_ensemble < 0, 1e-8, fold_test_ensemble)

        # 本折指标
        R2Score_train = r2_score(fold_train_reals, fold_train_ensemble)
        R2Score_trains.append(R2Score_train)
        R2Score_test = r2_score(fold_test_reals, fold_test_ensemble)
        R2Score_tests.append(R2Score_test)

        MSE_train = mean_squared_error(fold_train_reals, fold_train_ensemble)
        MSE_trains.append(MSE_train)
        MSE_test = mean_squared_error(fold_test_reals, fold_test_ensemble)
        MSE_tests.append(MSE_test)

        RMSE_train = math.sqrt(MSE_train)
        RMSE_trains.append(RMSE_train)
        RMSE_test = math.sqrt(MSE_test)
        RMSE_tests.append(RMSE_test)

        MAE_train = mean_absolute_error(fold_train_reals, fold_train_ensemble)
        MAE_trains.append(MAE_train)
        MAE_test = mean_absolute_error(fold_test_reals, fold_test_ensemble)
        MAE_tests.append(MAE_test)

        # 保存最后一折的详细预测（与原脚本行为接近）
        smiles_train = fold_smiles_train
        tems_train = fold_tems_train
        pres_train = fold_pres_train
        smiles_test = fold_smiles_test
        tems_test = fold_tems_test
        pres_test = fold_pres_test
        y_train_reals = fold_train_reals
        y_train_preds = fold_train_ensemble.tolist()
        y_test_reals = fold_test_reals
        y_test_preds = fold_test_ensemble.tolist()

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

    result_base = os.path.join('..', '..', 'data', 'result', normalized_target, 'Ensemble_Weighted')
    train_dir = os.path.join(result_base, 'train')
    test_dir = os.path.join(result_base, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    df_Score_summary = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score},
                                    index=['R2', 'MSE', 'RMSE', 'MAE'])
    print('Fold-wise test R2:', R2Score_tests)
    print('Summary scores:')
    print(df_Score_summary)
    summary_path = os.path.join(result_base, 'ensemble_Weighted.csv')
    # df_Score_summary.to_csv(summary_path)

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
    scores_path = os.path.join(result_base, 'ensemble_Weighted_score_5_fold.csv')
    # df_Score_all.to_csv(scores_path)


def parse_args():
     parser = argparse.ArgumentParser(
         description='Weighted ensemble evaluation for GNN models on IL datasets.'
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
     return parser.parse_args()


if __name__ == '__main__':
     args = parse_args()
     run_weighted_ensemble(args.target)
