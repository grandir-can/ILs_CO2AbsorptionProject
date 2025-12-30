import pandas as pd
import numpy as np
import os

def calculate_and_save_mean_scores(input_path, output_path):
    """
    Reads 5-fold scores from a given path, calculates the mean and standard deviation
    for each metric, and saves the formatted results to a new CSV file.

    Args:
        input_path (str): The path to the input CSV file ('5_fold_scores.csv').
        output_path (str): The path to save the output CSV file ('score_mean.csv').
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    metrics = ['R2', 'MSE', 'RMSE', 'MAE']
    train_scores = []
    test_scores = []

    for metric in metrics:
        train_col = f'Train {metric}'
        test_col = f'Test {metric}'

        # Calculate and format train scores
        train_mean = np.mean(df[train_col])
        train_std = np.std(df[train_col])
        train_scores.append(f'{round(train_mean, 4)}±{round(train_std, 4)}')

        # Calculate and format test scores
        test_mean = np.mean(df[test_col])
        test_std = np.std(df[test_col])
        test_scores.append(f'{round(test_mean, 4)}±{round(test_std, 4)}')

    # Create the results DataFrame
    mean_df_score = pd.DataFrame({
        'train_Score': train_scores,
        'test_Score': test_scores
    }, index=metrics)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a new CSV
    mean_df_score.to_csv(output_path)
    print(f"Mean scores saved to {output_path}")
    print("\n--- Calculated Mean Scores ---")
    print(mean_df_score)
    print("----------------------------")

if __name__ == '__main__':
    # Define file paths
    input_file = '../../data/result/CO2_capacity/XGBoost_RDKit/5_fold_scores.csv'
    output_file = '../../data/result/CO2_capacity/XGBoost_RDKit/score_mean.csv'

    # Run the calculation
    calculate_and_save_mean_scores(input_file, output_file)
