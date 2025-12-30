import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import os
import math
import pickle

def train_evaluate_xgboost(feature_type='RDKit', target_property='CO2_capacity'):
    # Select data and index paths based on target property
    if target_property == 'CO2_capacity':
        data_path = '../../data/CO2_capacity.xlsx'
        inds_path = '../../data/indexs/CO2_capacity/split_1'
        prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
    elif target_property == 'Viscosity':
        data_path = '../../data/viscosity.xlsx'
        inds_path = '../../data/indexs/viscosity'
        prop_col = 'log10(Viscosity,mPa.s)'
    else:
        raise ValueError("target_property must be 'CO2' or 'Viscosity'")

    df = pd.read_excel(data_path)
    smiles = df['smiles'].values
    props = df[prop_col].values
    tems = df['Temperature, K'].values
    pres = df['Pressure, kPa'].values

    # Scale temperature and pressure
    transfer1 = MinMaxScaler(feature_range=(0, 1))
    transfer2 = MinMaxScaler(feature_range=(0, 1))
    tems_scaled = transfer1.fit_transform(tems.reshape(-1, 1))
    pres_scaled = transfer2.fit_transform(pres.reshape(-1, 1))

    # Load pre-computed features
    if feature_type == 'RDKit':
        with open('../../data/rdkit_features.pkl', 'rb') as f:
            feature_dict = pickle.load(f)
    else:
        raise ValueError("feature_type must be 'RDKit'")

    all_features = np.array([feature_dict[s] for s in smiles])
    all_features = np.concatenate([all_features, tems_scaled, pres_scaled], axis=1)

    R2Score_trains, R2Score_tests = [], []
    MSE_trains, MSE_tests = [], []
    RMSE_trains, RMSE_tests = [], []
    MAE_trains, MAE_tests = [], []

    for n in range(5):
        print(n)
        train_index = pd.read_csv(os.path.join(inds_path, f'train_ind_{n}.csv'))['train_ind'].values
        test_index = pd.read_csv(os.path.join(inds_path, f'test_ind_{n}.csv'))['test_ind'].values

        X_train, X_test = all_features[train_index], all_features[test_index]
        y_train, y_test = props[train_index], props[test_index]

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='r2')
        grid_search.fit(X_train, y_train)

        # 输出最佳参数
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate and store metrics
        R2Score_trains.append(r2_score(y_train, y_train_pred))
        R2Score_tests.append(r2_score(y_test, y_test_pred))
        MSE_trains.append(mean_squared_error(y_train, y_train_pred))
        MSE_tests.append(mean_squared_error(y_test, y_test_pred))
        RMSE_trains.append(math.sqrt(mean_squared_error(y_train, y_train_pred)))
        RMSE_tests.append(math.sqrt(mean_squared_error(y_test, y_test_pred)))
        MAE_trains.append(mean_absolute_error(y_train, y_train_pred))
        MAE_tests.append(mean_absolute_error(y_test, y_test_pred))

        print(f'Fold {n+1} Test R2: {R2Score_tests[-1]:.4f}')

    # Print average scores
    print(f'Average Train R2: {np.mean(R2Score_trains):.4f} +- {np.std(R2Score_trains):.4f}')
    print(f'Average Test R2: {np.mean(R2Score_tests):.4f} +- {np.std(R2Score_tests):.4f}')

    # Save results
    results_df = pd.DataFrame({
        'Train R2': R2Score_trains,
        'Test R2': R2Score_tests,
        'Train MSE': MSE_trains,
        'Test MSE': MSE_tests,
        'Train RMSE': RMSE_trains,
        'Test RMSE': RMSE_tests,
        'Train MAE': MAE_trains,
        'Test MAE': MAE_tests
    })
    output_dir = f'../../data/result/{target_property}/XGBoost_{feature_type}/'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, '5_fold_scores.csv'), index=False)

    # Calculate and save average scores
    # Calculate and save average scores
    train_Score = [str(round(np.mean(R2Score_trains), 4)) + '+-' + str(round(np.std(R2Score_trains), 4)),
                   str(round(np.mean(MSE_trains), 4)) + '+-' + str(round(np.std(MSE_trains), 4)),
                   str(round(np.mean(RMSE_trains), 4)) + '+-' + str(round(np.std(RMSE_trains), 4)),
                   str(round(np.mean(MAE_trains), 4)) + '+-' + str(round(np.std(MAE_trains), 4))]
    test_Score = [str(round(np.mean(R2Score_tests), 4)) + '+-' + str(round(np.std(R2Score_tests), 4)),
                  str(round(np.mean(MSE_tests), 4)) + '+-' + str(round(np.std(MSE_tests), 4)),
                  str(round(np.mean(RMSE_tests), 4)) + '+-' + str(round(np.std(RMSE_tests), 4)),
                  str(round(np.mean(MAE_tests), 4)) + '+-' + str(round(np.std(MAE_tests), 4))]
    mean_df_Score = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score},
                                 index=['R2', 'MSE', 'RMSE', 'MAE'])
    print(mean_df_Score)
    avg_scores_df = pd.DataFrame(mean_df_Score)
    avg_scores_df.to_csv(os.path.join(output_dir, 'score_mean.csv'), index=False)

if __name__ == '__main__':
    train_evaluate_xgboost(feature_type='RDKit', target_property='CO2_capacity')
