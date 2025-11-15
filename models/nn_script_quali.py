"""
Neural Network Qualifying Predictions

Load trained neural network and generate qualifying predictions
for an upcoming race.

Usage:
    python nn_script_quali.py
"""

from pairwise_ranking_quali import NeuralNetsQuali, test_func, DataSetsQuali
import pandas as pd
import torch

def get_nn_quali_results(prediction_csv=None):
    """
    Get qualifying predictions from neural network

    Args:
        prediction_csv: Path to prediction CSV for upcoming race
                       If None, evaluates on test set
    """
    print("=" * 70)
    print("Neural Network Qualifying Predictions")
    print("=" * 70)

    # Load dataset
    if prediction_csv:
        print(f"\nLoading prediction data from: {prediction_csv}")
        d = DataSetsQuali(prediction_csv)
    else:
        print("\nLoading test dataset for evaluation...")
        d = DataSetsQuali("../data/HOLY_qualifying_v1.csv")

    # Load trained model
    print("\nLoading trained neural network...")
    model = NeuralNetsQuali(d.num_features)
    model.load_state_dict(torch.load("neural_nets_quali.pth"))
    model.eval()

    print(f"Model loaded: neural_nets_quali.pth")
    print(f"Features: {d.num_features}")

    # Generate predictions
    print("\nGenerating predictions...")

    if prediction_csv:
        # Predict for new data
        predictions = model(
            torch.tensor(d.testX.astype(float)).float()
        ).detach().numpy()

        # Add predictions to dataframe
        result_df = d.test_df_with_driver.copy()
        result_df['prediction'] = predictions

        # Sort by prediction (lower = better qualifying)
        result_df = result_df.sort_values('prediction')

        print("\n" + "=" * 70)
        print("PREDICTED QUALIFYING ORDER")
        print("=" * 70)

        for idx, (_, row) in enumerate(result_df.iterrows(), 1):
            driver = row['driver'] if 'driver' in row else f"Driver {idx}"
            pred_score = row['prediction']

            if idx == 1:
                print(f"P{idx:<3} {driver:<20} (Score: {pred_score:.4f})  🏆 POLE")
            elif idx <= 3:
                print(f"P{idx:<3} {driver:<20} (Score: {pred_score:.4f})  ⭐")
            elif idx <= 10:
                print(f"P{idx:<3} {driver:<20} (Score: {pred_score:.4f})  (Q3)")
            else:
                print(f"P{idx:<3} {driver:<20} (Score: {pred_score:.4f})")

        print("=" * 70)

        # Save results
        result_df[['driver', 'prediction']].to_csv('nn_quali_predictions.csv', index=False)
        print("\nPredictions saved to: nn_quali_predictions.csv")

    else:
        # Evaluate on test set
        test_predictions = model(
            torch.tensor(d.testX.astype(float)).float()
        ).detach().numpy()

        print("\nEvaluating model on test set (2019+)...")
        test_func(test_predictions, d.test_df_with_driver)

    return d


if __name__ == '__main__':
    # Option 1: Evaluate on test set
    get_nn_quali_results()

    # Option 2: Make predictions for upcoming race
    # Update the path to your prediction CSV
    # get_nn_quali_results(prediction_csv="../../prediction_dfs/pred_df_quali_monaco.csv")
