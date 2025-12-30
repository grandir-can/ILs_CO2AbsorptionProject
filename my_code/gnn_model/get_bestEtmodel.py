import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from util import smiles_to_data
from torch_geometric.data import DataLoader
from joblib import load
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import joblib

vis_data_path = '../../data/filter_data/viscosity.xlsx'
vis_df = pd.read_excel(vis_data_path)
vis_smiles = vis_df['smiles'].values
vis_cat_smiles = vis_df['cation smiles'].values.tolist()
vis_ani_smiles = vis_df['anion smiles'].values.tolist()
vis_props = vis_df['log10(Viscosity,mPa.s)'].values  # Mole_fraction_of_carbon_dioxide[Liquid]  log10(Viscosity,mPa.s) Heat Capacity,J/K/mol  Density,kg/m3
vis_tems = vis_df['Temperature, K'].values.reshape(-1, 1)
vis_pres = vis_df['Pressure, kPa'].values.reshape(-1, 1)

transfer3 = MinMaxScaler(feature_range=(0, 1))
transfer4 = MinMaxScaler(feature_range=(0, 1))
vis_tems = transfer3.fit_transform(vis_tems.reshape(-1, 1))
vis_pres = transfer4.fit_transform(vis_pres.reshape(-1, 1))

data = smiles_to_data(vis_smiles, vis_tems, vis_pres,vis_props, 'single')

data_loader = DataLoader(data, batch_size=128, shuffle=False)

device = torch.device('cuda:2')

model_GIN = load('../../model/viscosity/GIN/all/model.pt').to(device)
model_GAT = load('../../model/viscosity/GAT/all/model.pt').to(device)
model_PNA = load('../../model/viscosity/PNA/all/model.pt').to(device)
model_MPNN = load('../../model/viscosity/MPNN/all/model.pt').to(device)
model_Attentive_FP = load('../../model/viscosity/Attentive_FP/all/model_2.pt').to(device)

y_train_real_all = []
y_train_pred_GIN_all,y_train_pred_PNA_all,y_train_pred_GAT_all,y_train_pred_MPNN_all,y_train_pred_Attentive_FP_all = [],[],[],[],[]
for d in data_loader:
    d = d.to(device)
    model_Attentive_FP.eval()
    model_GIN.eval()
    model_GAT.eval()
    model_MPNN.eval()
    model_PNA.eval()
    with torch.no_grad():
        y_train_pred_GIN = model_GIN(d).cpu().detach().numpy().tolist()
        y_train_pred_PNA = model_PNA(d).cpu().detach().numpy().tolist()
        y_train_pred_GAT = model_GAT(d).cpu().detach().numpy().reshape(-1).tolist()
        y_train_pred_MPNN = model_MPNN(d).cpu().detach().numpy().tolist()
        y_train_pred_Attentive_FP = model_Attentive_FP(d).cpu().detach().numpy().tolist()

        y_train_pred_GIN_all.extend(y_train_pred_GIN)
        y_train_pred_PNA_all.extend(y_train_pred_PNA)
        y_train_pred_GAT_all.extend(y_train_pred_GAT)
        y_train_pred_MPNN_all.extend(y_train_pred_MPNN)
        y_train_pred_Attentive_FP_all.extend(y_train_pred_Attentive_FP)

        y_train_real = d.y.cpu().numpy().tolist()
        y_train_real_all.extend(y_train_real)

stacked_X_train = np.column_stack((
    np.array(y_train_pred_GIN_all),
    np.array(y_train_pred_PNA_all),
    np.array(y_train_pred_GAT_all),
    np.array(y_train_pred_MPNN_all),
    np.array(y_train_pred_Attentive_FP_all)))

# 定义参数网格 ExtraTreesRegressor
param_grid = {
    'n_estimators': [100, 200, 400, 600, 800],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [1, 2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
# 初始化模型
meta = ExtraTreesRegressor(random_state=42)

# 网格搜索
grid_search = GridSearchCV(estimator=meta, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(stacked_X_train, y_train_real_all)
# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# ========== 保存最优模型 ==========
# grid_search.best_estimator_ 是网格搜索后得到的最优模型实例
best_model = grid_search.best_estimator_
# 保存路径（可自定义，比如保存为 .pkl 或 .joblib 格式）
save_path = "../../model/viscosity/best_ensemble_stack_ET_model.joblib"
joblib.dump(best_model, save_path)
print(f"最优模型已保存至: {save_path}")