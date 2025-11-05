"""
Neural Network for Qualifying Position Prediction

This model uses pairwise ranking loss to learn qualifying order.
It's designed specifically for predicting grid positions (qualifying results).

Target: grid (qualifying position)
Dataset: HOLY_qualifying_v1.csv
Architecture: Input -> 60 -> 30 -> 1 (pairwise ranking scores)
"""

import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)


class NeuralNetsQuali(nn.Module):
    '''
    Neural network for qualifying prediction
    '''

    def __init__(self, num_features) -> None:
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

    def forward(self, x):
        res = self.linear_relu_stack(x)
        return res


class DataSetsQuali():
    '''
    Dataset loader for qualifying data
    '''

    def __init__(self, csv_file):
        print(f"Loading qualifying dataset from: {csv_file}")
        df = pd.read_csv(csv_file)

        # Store test dataframe with driver names for evaluation
        self.test_df_with_driver = df[df['season'] > 2018].reset_index(drop=True)

        # Drop unnecessary columns
        if 'driver' in df.columns:
            df = df.drop(columns=['driver'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Drop circuit and nationality features (too many columns, may not help)
        df = df.drop(columns=df.columns[df.columns.str.startswith('circuit_id')], errors='ignore')
        df = df.drop(columns=df.columns[df.columns.str.startswith('nationality')], errors='ignore')

        # Split data temporally
        self.train_df = df[df["season"] < 2014].reset_index(drop=True)
        self.val_df = df[(df['season'] >= 2014) & (df['season'] <= 2018)].reset_index(drop=True)
        self.test_df = df[df['season'] > 2018].reset_index(drop=True)

        # Separate features and target (grid instead of podium)
        self.trainX = self.train_df.drop(columns=['grid'])
        self.trainy = self.train_df['grid']

        self.valX = self.val_df.drop(columns=['grid'])
        self.valy = self.val_df['grid']

        self.testX = self.test_df.drop(columns=['grid'])
        self.testy = self.test_df['grid']

        # EXPERIMENT: Drop qualifying_secs to test model without lap time data
        if 'qualifying_secs' in self.trainX.columns:
            print(f"  🔬 EXPERIMENT: Dropping 'qualifying_secs' column")
            self.trainX = self.trainX.drop(columns=['qualifying_secs'])
            self.valX = self.valX.drop(columns=['qualifying_secs'])
            self.testX = self.testX.drop(columns=['qualifying_secs'])

        # Drop object columns (strings that can't be converted to float)
        object_cols = self.trainX.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)}")
            self.trainX = self.trainX.drop(columns=object_cols)
            self.valX = self.valX.drop(columns=object_cols)
            self.testX = self.testX.drop(columns=object_cols)

        # Convert boolean columns to int
        bool_cols = self.trainX.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            print(f"  Converting {len(bool_cols)} boolean columns to int")
            self.trainX[bool_cols] = self.trainX[bool_cols].astype(int)
            self.valX[bool_cols] = self.valX[bool_cols].astype(int)
            self.testX[bool_cols] = self.testX[bool_cols].astype(int)

        # Save number of features before converting to numpy
        self.num_features = len(self.trainX.columns)

        # Standardize features
        self.scaler = StandardScaler()
        # Convert to numpy arrays for proper tensor conversion
        self.trainX = self.scaler.fit_transform(self.trainX.values)
        self.valX = self.scaler.transform(self.valX.values)
        self.testX = self.scaler.transform(self.testX.values)

        print(f"Training set: {len(self.trainX)} records (pre-2014)")
        print(f"Validation set: {len(self.valX)} records (2014-2018)")
        print(f"Test set: {len(self.testX)} records (2019+)")
        print(f"Number of features: {self.num_features}")


def loss_func(pred, labels, df, margin=5):
    '''
    Pairwise ranking loss function for qualifying positions

    For each qualifying session, drivers who qualified better (lower grid number)
    should have lower prediction scores than drivers who qualified worse.

    Args:
        pred: Model predictions (lower = better qualifying)
        labels: Actual qualifying positions (grid)
        df: Dataframe with season/round info
        margin: Minimum score difference required between drivers
    '''
    total_loss = 0
    count = 0

    # Group by qualifying session (season + round)
    for (season, round_num), group_idx in df.groupby(['season', 'round']).groups.items():
        group_labels = labels.iloc[group_idx].values
        group_pred = pred[group_idx]

        # For each pair of drivers in this session
        for i in range(len(group_idx)):
            for j in range(i + 1, len(group_idx)):
                # Driver i qualified better than driver j if grid_i < grid_j
                if group_labels[i] < group_labels[j]:
                    # pred[i] should be < pred[j]
                    loss = torch.max(torch.tensor(0.0), group_pred[j] + margin - group_pred[i])
                    total_loss += loss
                    count += 1
                elif group_labels[i] > group_labels[j]:
                    # pred[j] should be < pred[i]
                    loss = torch.max(torch.tensor(0.0), group_pred[i] + margin - group_pred[j])
                    total_loss += loss
                    count += 1

    return total_loss / count if count > 0 else total_loss


def test_func(pred, test_df_with_driver):
    '''
    Evaluate qualifying prediction accuracy

    Metrics:
    - Pole position accuracy
    - Top 2 in order
    - Top 3 in order
    - Top 3 any order
    '''
    test_df_with_driver['pred'] = pred

    num_correct_pole = 0
    num_correct_second = 0
    num_correct_third = 0
    num_correct_top_two = 0
    num_correct_top_three_order = 0
    num_correct_top_three = 0

    num_total_indiv = 0
    num_total_sessions = 0

    for (season, round_num), group_idx in test_df_with_driver.groupby(['season', 'round']).groups.items():
        race_df = test_df_with_driver.iloc[group_idx]

        # Skip if no data
        if len(race_df) == 0:
            continue

        # Predicted top 3 (lowest prediction scores)
        top_pred_id = race_df.sort_values(by=['pred'], ascending=True).iloc[:3]

        # Actual top 3 (grid positions 1, 2, 3)
        real_top_id = race_df.sort_values(by=['grid'], ascending=True).iloc[:3]

        # Pole position accuracy
        if len(top_pred_id) > 0 and len(real_top_id) > 0:
            num_correct_pole += top_pred_id.iloc[0].name == real_top_id.iloc[0].name

        # 2nd place accuracy
        if len(top_pred_id) > 1 and len(real_top_id) > 1:
            num_correct_second += top_pred_id.iloc[1].name == real_top_id.iloc[1].name

        # 3rd place accuracy
        if len(top_pred_id) > 2 and len(real_top_id) > 2:
            num_correct_third += top_pred_id.iloc[2].name == real_top_id.iloc[2].name

        # Top 2 in order
        if len(top_pred_id) > 1 and len(real_top_id) > 1:
            num_correct_top_two += (
                (top_pred_id.iloc[0].name == real_top_id.iloc[0].name) and
                (top_pred_id.iloc[1].name == real_top_id.iloc[1].name)
            )

        # Top 3 in exact order
        if len(top_pred_id) > 2 and len(real_top_id) > 2:
            num_correct_top_three_order += (
                (top_pred_id.iloc[0].name == real_top_id.iloc[0].name) and
                (top_pred_id.iloc[1].name == real_top_id.iloc[1].name) and
                (top_pred_id.iloc[2].name == real_top_id.iloc[2].name)
            )

        # Top 3 in any order
        top_3_pred = set(race_df.sort_values(by=['pred'], ascending=True).iloc[:3].index)
        real_3 = set(race_df.sort_values(by=['grid'], ascending=True).iloc[:3].index)
        num_correct_top_three += top_3_pred == real_3

        num_total_indiv += 1
        num_total_sessions += 1

    results = {
        'pole_only': num_correct_pole / num_total_indiv if num_total_indiv > 0 else 0,
        'second_only': num_correct_second / num_total_indiv if num_total_indiv > 0 else 0,
        'third_only': num_correct_third / num_total_indiv if num_total_indiv > 0 else 0,
        'top_two_in_order': num_correct_top_two / num_total_indiv if num_total_indiv > 0 else 0,
        'top_three_in_order': num_correct_top_three_order / num_total_indiv if num_total_indiv > 0 else 0,
        'top_three_any_order': num_correct_top_three / num_total_sessions if num_total_sessions > 0 else 0
    }

    print("\n" + "=" * 70)
    print("QUALIFYING PREDICTION ACCURACY")
    print("=" * 70)
    print(f"Pole Position Accuracy: {results['pole_only']*100:.2f}%")
    print(f"2nd Place Accuracy: {results['second_only']*100:.2f}%")
    print(f"3rd Place Accuracy: {results['third_only']*100:.2f}%")
    print(f"Top 2 in Order: {results['top_two_in_order']*100:.2f}%")
    print(f"Top 3 in Order: {results['top_three_in_order']*100:.2f}%")
    print(f"Top 3 Any Order: {results['top_three_any_order']*100:.2f}%")
    print("=" * 70)

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("Neural Network Qualifying Prediction Training")
    print("=" * 70)

    # Load data
    d = DataSetsQuali('../data/HOLY_qualifying_v1.csv')

    epochs = 100
    model_path = "neural_nets_quali.pth"

    # Training mode (set to False to load existing model)
    TRAIN_NEW_MODEL = True

    if not TRAIN_NEW_MODEL:
        print(f"\nLoading existing model from: {model_path}")
        model = NeuralNetsQuali(d.num_features)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("\nTraining new model...")
        model = NeuralNetsQuali(d.num_features)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

        losses = []
        val_losses = []

        print(f"Training for {epochs} epochs with early stopping...")

        for i in tqdm(range(epochs)):
            # Training
            optimizer.zero_grad()
            pred = model(torch.tensor(d.trainX.astype(float)).float())
            loss = loss_func(pred, d.trainy, d.train_df)

            # Validation
            val_pred = model(torch.tensor(d.valX.astype(float)).float())

            # Early stopping check
            num_prev = 8
            if i > num_prev:
                avg = sum(val_losses[-num_prev:]) / num_prev
                val_loss = loss_func(val_pred, d.valy, d.val_df)

                if val_loss.detach().numpy() > avg:
                    print(f"\nEarly stopping at epoch {i}")
                    break

            val_loss = loss_func(val_pred, d.valy, d.val_df)

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().numpy())
            val_losses.append(val_loss.detach().numpy())

        print(f"\nTraining complete! Final loss: {float(losses[-1]):.4f}")

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Evaluate on test set
    print("\nEvaluating on test set (2019+)...")
    model.eval()
    test_pred = model(torch.tensor(d.testX.astype(float)).float()).detach().numpy()
    results = test_func(test_pred, d.test_df_with_driver)

    print("\nTraining complete! Use nn_script_quali.py to make predictions.")
