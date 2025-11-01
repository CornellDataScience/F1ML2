"""
XGBoost Classifier for Qualifying Position Prediction

This model classifies drivers into qualifying groups:
- Class 1: Pole position
- Class 2: 2nd place
- Class 3: 3rd place
- Class 0: Outside top 3

Target: grid (qualifying position)
Dataset: HOLY_qualifying_v1.csv
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

def train(training_csv_path):
    """
    Train XGBoost classifier for qualifying predictions
    """
    print("=" * 70)
    print("Training XGBoost Classifier for Qualifying Prediction")
    print("=" * 70)

    # Load data
    print(f"\nLoading training data from: {training_csv_path}")
    df = pd.read_csv(training_csv_path)

    # Drop unnecessary columns
    cols_to_drop = ['driver']
    if 'Unnamed: 0' in df.columns:
        cols_to_drop.append('Unnamed: 0')

    df_enc = df.drop(cols_to_drop, axis=1, errors='ignore')

    # EXPERIMENT: Drop qualifying_secs to test model without lap time data
    if 'qualifying_secs' in df_enc.columns:
        print(f"  🔬 EXPERIMENT: Dropping 'qualifying_secs' column")
        df_enc = df_enc.drop(columns=['qualifying_secs'])

    # Drop any remaining object/string columns
    object_cols = df_enc.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)}")
        df_enc = df_enc.drop(columns=object_cols)

    # Convert weather and boolean columns to int
    weather_cols = ['weather_warm', 'weather_cold', 'weather_dry', 'weather_wet', 'weather_cloudy']
    for col in weather_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype(int)

    bool_cols = df_enc.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"  Converting {len(bool_cols)} boolean columns to int")
        df_enc[bool_cols] = df_enc[bool_cols].astype(int)

    print(f"Dataset shape: {df_enc.shape}")
    print(f"Date range: {df_enc['season'].min()}-{df_enc['season'].max()}")

    # Split train/test by season
    print("\nSplitting data: <=2018 for training, >=2019 for testing")
    y_train = df_enc.loc[df_enc['season'] <= 2018][['grid']]
    X_train = df_enc.loc[df_enc['season'] <= 2018].drop(['grid'], axis=1)
    y_test = df_enc.loc[df_enc['season'] >= 2019][['grid']]
    X_test = df_enc.loc[df_enc['season'] >= 2019].drop(['grid'], axis=1)

    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")

    # Convert qualifying positions to classes
    # Pole = 1, 2nd = 2, 3rd = 3, Others = 0
    print("\nConverting qualifying positions to classes...")
    print("  Class 1: Pole position")
    print("  Class 2: 2nd place")
    print("  Class 3: 3rd place")
    print("  Class 0: Outside top 3")

    for row in y_train.iterrows():
        if row[1][0] > 3:
            row[1][0] = 0

    for row in y_test.iterrows():
        if row[1][0] > 3:
            row[1][0] = 0

    # Train XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    best_max_depth = 3
    best_gamma = 1
    best_eta = 0.6187849412845402

    xgb_cl = xgb.XGBClassifier(
        gamma=best_gamma,
        max_depth=best_max_depth,
        eta=best_eta,
        eval_metric='mlogloss'
    )

    xgb_cl.fit(X_train, y_train)
    print("Training complete!")

    # Get probabilities
    print("\nGenerating predictions on test set...")
    probs = xgb_cl.predict_proba(X_test)

    # Normalize probabilities
    probs = probs / np.sum(probs, axis=0)

    # Calculate accuracy
    actual = X_test.copy()
    actual['grid'] = y_test

    accuracyDF = actual[['season', 'round', 'grid']]
    places = ["pole", "second", "third"]
    i = 1
    for place in places:
        accuracyDF[place + "_place_probs"] = probs[:, i]
        actual[place + "_place_probs"] = probs[:, i]
        i += 1

    accuracyDF['placement_prediction'] = 0

    # Make predictions by qualifying session
    def make_predictions(group):
        group['placement_prediction'] = 0

        # Pole position (highest pole probability)
        max_pole_prob = group['pole_place_probs'].max()
        pole_mask = group['pole_place_probs'] == max_pole_prob
        group.loc[pole_mask, 'placement_prediction'] = 1

        # 2nd place
        remaining = group[~pole_mask]
        if len(remaining) > 0:
            max_second_prob = remaining['second_place_probs'].max()
            second_mask = group['second_place_probs'] == max_second_prob
            group.loc[second_mask & ~pole_mask, 'placement_prediction'] = 2

            # 3rd place
            remaining2 = remaining[~second_mask]
            if len(remaining2) > 0:
                max_third_prob = remaining2['third_place_probs'].max()
                third_mask = group['third_place_probs'] == max_third_prob
                group.loc[third_mask & ~pole_mask & ~second_mask, 'placement_prediction'] = 3

        return group

    # Apply predictions per session
    accuracyDF = accuracyDF.groupby(['season', 'round']).apply(make_predictions).reset_index(drop=True)

    # Calculate pole position accuracy
    pole_predictions = accuracyDF[accuracyDF['placement_prediction'] == 1]
    pole_correct = (pole_predictions['grid'] == 1).sum()
    pole_total = len(pole_predictions)
    pole_accuracy = (pole_correct / pole_total * 100) if pole_total > 0 else 0

    # Calculate top 3 accuracy
    top3_predictions = accuracyDF[accuracyDF['placement_prediction'].isin([1, 2, 3])]
    top3_correct = (top3_predictions['grid'] == top3_predictions['placement_prediction']).sum()
    top3_total = len(top3_predictions)
    top3_accuracy = (top3_correct / top3_total * 100) if top3_total > 0 else 0

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE ON TEST SET (2019-2023)")
    print("=" * 70)
    print(f"Pole Position Accuracy: {pole_accuracy:.2f}% ({pole_correct}/{pole_total})")
    print(f"Top 3 Position Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{top3_total})")
    print("=" * 70)

    # Save model
    model_filename = "xgb_quali_classifier.joblib"
    joblib.dump(xgb_cl, model_filename)
    print(f"\nModel saved to: {model_filename}")

    return xgb_cl, accuracyDF


def run(predictions_csv_path, race_name="upcoming_quali"):
    """
    Run predictions on a new qualifying session
    """
    print("\n" + "=" * 70)
    print(f"Generating Qualifying Predictions for: {race_name}")
    print("=" * 70)

    # Load trained model
    print("\nLoading trained model...")
    model = joblib.load("xgb_quali_classifier.joblib")
    print("Model loaded: xgb_quali_classifier.joblib")

    # Load prediction data
    print(f"\nLoading prediction data from: {predictions_csv_path}")
    predictionDF = pd.read_csv(predictions_csv_path)

    # Drop driver column for prediction
    driver_names = predictionDF['driver'].copy() if 'driver' in predictionDF.columns else None

    cols_to_drop = ['driver']
    if 'Unnamed: 0' in predictionDF.columns:
        cols_to_drop.append('Unnamed: 0')
    if 'grid' in predictionDF.columns:
        cols_to_drop.append('grid')

    X_pred = predictionDF.drop(cols_to_drop, axis=1, errors='ignore')

    # EXPERIMENT: Drop qualifying_secs to test model without lap time data
    if 'qualifying_secs' in X_pred.columns:
        print(f"  🔬 EXPERIMENT: Dropping 'qualifying_secs' column")
        X_pred = X_pred.drop(columns=['qualifying_secs'])

    # Drop any remaining object/string columns
    object_cols = X_pred.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)}")
        X_pred = X_pred.drop(columns=object_cols)

    # Convert weather and boolean columns to int
    weather_cols = ['weather_warm', 'weather_cold', 'weather_dry', 'weather_wet', 'weather_cloudy']
    for col in weather_cols:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype(int)

    bool_cols = X_pred.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X_pred[bool_cols] = X_pred[bool_cols].astype(int)

    # Make predictions
    print("\nGenerating predictions...")
    probs = model.predict_proba(X_pred)
    probs = probs / np.sum(probs, axis=0)

    # Add probabilities to dataframe
    if driver_names is not None:
        predictionDF = pd.DataFrame({'driver': driver_names})

    predictionDF['pole_place_probs'] = probs[:, 1]
    predictionDF['second_place_probs'] = probs[:, 2]
    predictionDF['third_place_probs'] = probs[:, 3]

    # Find predictions
    pole_idx = predictionDF['pole_place_probs'].idxmax()
    pole_prediction = predictionDF.loc[pole_idx, 'driver'] if 'driver' in predictionDF.columns else f"Driver {pole_idx}"

    remaining = predictionDF[predictionDF.index != pole_idx]
    second_idx = remaining['second_place_probs'].idxmax()
    second_prediction = predictionDF.loc[second_idx, 'driver'] if 'driver' in predictionDF.columns else f"Driver {second_idx}"

    remaining2 = remaining[remaining.index != second_idx]
    third_idx = remaining2['third_place_probs'].idxmax()
    third_prediction = predictionDF.loc[third_idx, 'driver'] if 'driver' in predictionDF.columns else f"Driver {third_idx}"

    # Display predictions
    print("\n" + "=" * 70)
    print("PREDICTED TOP 3 QUALIFYING POSITIONS")
    print("=" * 70)
    print(f"🏆 Pole Position: {pole_prediction}")
    print(f"⭐ 2nd Place: {second_prediction}")
    print(f"⭐ 3rd Place: {third_prediction}")
    print("=" * 70)

    # Save predictions
    output_file = f"xgb_quali_predictions_for_{race_name}.csv"
    predictionDF.to_csv(output_file, index=False)
    print(f"\nFull predictions saved to: {output_file}")

    return {
        "Pole": pole_prediction,
        "Second": second_prediction,
        "Third": third_prediction
    }


# CONFIGURATION - MODIFY THESE PATHS
if __name__ == "__main__":
    training_csv_path = "../data/HOLY_qualifying_v1.csv"
    predictions_csv_path = "../../prediction_dfs/pred_df_quali_monaco.csv"
    race_name = "monaco"

    # Train the model
    print("Step 1: Training model on historical qualifying data")
    print("You can skip this if model already exists (comment out the train() call)")
    train(training_csv_path)

    # Make predictions (comment out if you don't have a prediction CSV yet)
    # print("\n\nStep 2: Making predictions for upcoming race")
    # run(predictions_csv_path, race_name)
