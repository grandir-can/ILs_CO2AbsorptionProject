import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from util import smiles_to_data
from gnn import GAT, GIN, PNA, MPNN, AttentiveFP


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified training script for GNN models on IL datasets.'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='GNN model to train: MPNN, GAT, GIN, PNA, AttentiveFP (case-insensitive).'
    )
    parser.add_argument(
        '--target', type=str, default='CO2_capacity',
        help='Target property: "CO2_capacity" or "viscosity" (case-insensitive).'
    )
    parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of CV folds (default: 5, requires precomputed index files).'
    )
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Number of training epochs per fold. If not set, choose a default per model.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Batch size (default: 128).'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate. If not set, choose a default per model.'
    )
    parser.add_argument(
        '--data-dir', type=str, default='../../data',
        help='Base data directory (default: ../../data).'
    )
    parser.add_argument(
        '--index-base', type=str, default='../../data/indexs',
        help='Base index directory containing CO2_capacity/ and viscosity/ (default: ../../data/indexs).'
    )
    parser.add_argument(
        '--model-base', type=str, default='../../model',
        help='Base directory to save trained models (default: ../../model).'
    )
    parser.add_argument(
        '--log-base', type=str, default='../../data/epoch_loss',
        help='Base directory to save epoch loss CSVs (default: ../../data/epoch_loss).'
    )
    return parser.parse_args()


def _normalize_target(target: str) -> str:
    key = target.strip().lower()
    if key in ('co2_capacity', 'co2', 'co2capacity'):
        return 'CO2_capacity'
    if key in ('viscosity', 'vis'):
        return 'viscosity'
    raise ValueError(f'Unsupported target property: {target}. Use "CO2_capacity" or "viscosity".')


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def prepare_dataset(args, normalized_target: str):
    """Load data and return (data_list, indices_dir)."""
    if normalized_target == 'CO2_capacity':
        data_path = os.path.join(args.data_dir, 'CO2_capacity.xlsx')
        prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
        indices_dir = os.path.join(args.index_base, 'CO2_capacity')
    else:  # viscosity
        data_path = os.path.join(args.data_dir, 'viscosity.xlsx')
        prop_col = 'log10(Viscosity,mPa.s)'
        indices_dir = os.path.join(args.index_base, 'viscosity')

    df = pd.read_excel(data_path)
    smiles = df['smiles'].values
    props = df[prop_col].values
    tems = df['Temperature, K'].values
    pres = df['Pressure, kPa'].values

    transfer1 = MinMaxScaler(feature_range=(0, 1))
    transfer2 = MinMaxScaler(feature_range=(0, 1))
    tems = transfer1.fit_transform(tems.reshape(-1, 1))
    pres = transfer2.fit_transform(pres.reshape(-1, 1))

    data_list = smiles_to_data(smiles, tems, pres, props, 'single',args.target)
    return data_list, indices_dir


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_all = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.y.to(torch.float32)
        if out.ndim > 1:
            out = out.view(-1)
        loss = F.mse_loss(out, target)
        loss.backward()
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


def evaluate_mse(model, loader, device):
    model.eval()
    error = 0.0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
        target = batch.y.to(torch.float32)
        if out.ndim > 1:
            out = out.view(-1)
        error += mean_squared_error(target.detach().cpu().numpy(),
                                    out.detach().cpu().numpy())
    return error / len(loader.dataset)


def compute_pna_deg(train_loader):
    """Compute degree histogram tensor for PNA from a training DataLoader."""
    max_degree = -1
    for batch in train_loader:
        d = degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for batch in train_loader:
        d = degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


def build_model(model_key: str, in_channels: int, device: torch.device, pna_deg=None):
    """Construct a GNN model from gnn.py according to model_key."""
    if model_key == 'mpnn':
        model = MPNN(in_channels=in_channels, hidden_channels=128, out_channels=1)
    elif model_key == 'gat':
        model = GAT(in_channels=in_channels, hidden_channels1=100, hidden_channels2=64, out_channels=1)
    elif model_key == 'gin':
        model = GIN(in_channels=in_channels, hidden_channels1=128, hidden_channels2=64, out_channels=1)
    elif model_key == 'pna':
        if pna_deg is None:
            raise ValueError('pna_deg (degree histogram) must be provided for PNA model.')
        model = PNA(in_channels=in_channels, hidden_channels=100, out_channels=1, deg=pna_deg)
    elif model_key in ('attentivefp', 'attentive_fp', 'attentive'):
        model = AttentiveFP(in_channels=in_channels, hidden_channels=200, out_channels=1,
                            edge_dim=6, num_layers=3, num_timesteps=3, dropout=0.4)
    else:
        raise ValueError(f'Unsupported GNN model: {model_key}')
    return model.to(device)


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalized_target = _normalize_target(args.target)
    model_key = _normalize_model_name(args.model)

    data_list, indices_dir = prepare_dataset(args, normalized_target)
    if not data_list:
        raise RuntimeError('No data was loaded from the dataset.')

    in_channels = data_list[0].x.size(-1)

    # 依据模型类型选择默认超参数
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 1e-4 if model_key == 'pna' else 1e-3

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 500 if model_key == 'pna' else 200

    # 输出路径：与原有结构兼容
    if model_key in ('attentivefp', 'attentive_fp', 'attentive'):
        model_folder = 'Attentive_FP'
    else:
        model_folder = model_key.upper()

    model_dir = os.path.join(args.model_base, normalized_target, model_folder, 'split_1', 'single2')
    log_dir = os.path.join(args.log_base, normalized_target, model_folder, 'split_1', 'single2')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    start = time.perf_counter()

    for fold in range(args.folds):
        train_index_path = os.path.join(indices_dir, f'train_ind_{fold}.csv')
        test_index_path = os.path.join(indices_dir, f'test_ind_{fold}.csv')

        if not os.path.exists(train_index_path) or not os.path.exists(test_index_path):
            raise FileNotFoundError(
                f'Missing index files for fold {fold}: {train_index_path} or {test_index_path}'
            )

        train_index = pd.read_csv(train_index_path)['train_ind'].values
        test_index = pd.read_csv(test_index_path)['test_ind'].values

        train_data = [data_list[idx] for idx in train_index]
        test_data = [data_list[idx] for idx in test_index]

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        pna_deg = None
        if model_key == 'pna':
            pna_deg = compute_pna_deg(train_loader)

        model = build_model(model_key, in_channels, device, pna_deg=pna_deg)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5
        )

        epochs_list, train_losses, test_errors = [], [], []

        for epoch in range(1, epochs + 1):
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step(train_loss)
            test_error = evaluate_mse(model, test_loader, device)

            epochs_list.append(epoch)
            train_losses.append(train_loss)
            test_errors.append(test_error)

            print(
                f'Model: {model_key}, Target: {normalized_target}, Fold: {fold}, '
                f'Epoch: {epoch:03d}, LR: {current_lr:.7f}, '
                f'Train Loss: {train_loss:.7f}, Test MSE: {test_error:.7f}'
            )

        final_model_path = os.path.join(model_dir, f'model_{fold}.pt')
        dump(model, final_model_path)

        df_metrics = pd.DataFrame({
            'epoch': epochs_list,
            'train_loss': train_losses,
            'test_mse': test_errors,
        })
        metrics_path = os.path.join(log_dir, f'epoch_loss_{fold}.csv')
        df_metrics.to_csv(metrics_path, index=False)

    end = time.perf_counter()
    print(f'Average time per fold: {(end - start) / max(1, args.folds):.2f} seconds')


if __name__ == '__main__':
    main()
