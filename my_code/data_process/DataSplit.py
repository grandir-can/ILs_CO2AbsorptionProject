import numpy as np
import pandas as pd
import os
import argparse

def group_fold_split_data(key,k, index_save_path, data_df):
    groups = data_df[key]

    unique_groups, groups = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    n_samples_per_group = np.bincount(groups)
    group_to_fold = np.zeros(len(unique_groups))


    indxs = np.arange(n_groups)
    num = k
    count = n_groups // k

    for i in range(count):
        print(i)
        z = np.random.choice(indxs, num, False)
        for j in range(num):
            index = np.where(indxs == z[j])
            indxs = np.delete(indxs, index)
        if i == 0:
            sort_index = np.argsort(n_samples_per_group[z])[::-1]
            z = z[sort_index]
            n_samples_per_fold = n_samples_per_group[z]
            group_to_fold[z] = range(k)

        else:

            sort_index = np.argsort(n_samples_per_group[z])
            z = z[sort_index]
            n_samples_per_group_z= n_samples_per_group[z]

            fold_index = np.argsort(n_samples_per_fold)[::-1]

            n =0
            for ind in fold_index:
                n_samples_per_fold[ind] = n_samples_per_fold[ind] + n_samples_per_group_z[n]
                n = n+1
            group_to_fold[z] = fold_index
    for id in indxs:
        lightest_fold = np.argmin(n_samples_per_fold)
        n_samples_per_fold[lightest_fold] += n_samples_per_group[id]
        group_to_fold[id] = lightest_fold


    indices = group_to_fold[groups]

    for f in range(k):
        train = np.where(indices != f)[0]
        test = np.where(indices == f)[0]
        train_ind = pd.DataFrame(train)
        train_ind.columns = ['train_ind']
        # train_ind.to_csv(os.path.join(index_save_path, 'train_ind_' + str(f) + '.csv'), index=False)

        test_ind = pd.DataFrame(test)
        test_ind.columns = ['test_ind']
        # test_ind.to_csv(os.path.join(index_save_path, 'test_ind_' + str(f) + '.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group-based k-fold data splitting for ILs datasets.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to split: "CO2_capacity" or "viscosity".')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for group-based k-fold splitting (default: 5).')
    parser.add_argument('--group-key', type=str, default='smiles',
                        help='Column name used to define groups for splitting (default: "smiles").')
    parser.add_argument('--index-dir', type=str, default=None,
                        help='Directory to save train/test index CSV files. '
                             'If not provided, a default path is chosen based on the dataset.')

    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')

    dataset_name = args.dataset.lower()
    if dataset_name in ['co2_capacity', 'co2', 'co2capacity']:
        data_path = os.path.join(data_dir, 'CO2_capacity.xlsx')
        default_index_dir = os.path.join(project_root, 'data', 'indexs', 'CO2_capacity')
        prop_col = 'Mole_fraction_of_carbon_dioxide[Liquid]'
    elif dataset_name in ['viscosity', 'vis']:
        data_path = os.path.join(data_dir, 'viscosity.xlsx')
        default_index_dir = os.path.join(project_root, 'data', 'indexs', 'viscosity')
        prop_col = 'log10(Viscosity,mPa.s)'
    else:
        raise ValueError('Unsupported dataset: {}. Use "CO2_capacity" or "viscosity".'.format(args.dataset))

    index_save_path = args.index_dir if args.index_dir is not None else default_index_dir
    os.makedirs(index_save_path, exist_ok=True)

    df = pd.read_excel(data_path)
    props = df[prop_col].values.tolist()

    group_fold_split_data(args.group_key, args.k, index_save_path, df)
    print("success!")