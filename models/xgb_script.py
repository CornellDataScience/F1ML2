import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy
import sklearn
import scipy.stats as ss
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hyperopt import Trials
from scipy.optimize import fmin
import joblib
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
import os
import numpy as np

# given a dataset saves the model as xgb_classifier.joblib, saves the prediction DF as "xgb_boost_predictions.csv"


def train(training_csv_path):

    # retrieve data, data manipulation
    df = pd.read_csv(training_csv_path)
    df_enc = df.drop(['driver', 'Unnamed: 0'], axis=1)
    df_enc['weather_warm'] = df_enc['weather_warm'].astype(int)
    df_enc['weather_cold'] = df_enc['weather_cold'].astype(int)
    df_enc['weather_dry'] = df_enc['weather_dry'].astype(int)
    df_enc['weather_wet'] = df_enc['weather_wet'].astype(int)
    df_enc['weather_cloudy'] = df_enc['weather_cloudy'].astype(int)

    y_train = df_enc.loc[df_enc['season'] <= 2018][['podium']]
    X_train = df_enc.loc[df_enc['season'] <= 2018].drop(['podium'], axis=1)
    y_test = df_enc.loc[df_enc['season'] >= 2019][['podium']]
    X_test = df_enc.loc[df_enc['season'] >= 2019].drop(['podium'], axis=1)

    for row in y_train.iterrows():
        if (row[1][0] > 3):
            row[1][0] = 0

    for row in y_test.iterrows():
        if (row[1][0] > 3):
            row[1][0] = 0

    for row in y_train.iterrows():
        if (row[1][0] > 3):
            row[1][0] = 0

    for row in y_test.iterrows():
        if (row[1][0] > 3):
            row[1][0] = 0

    # run the actual model
    best_max_depth = 3
    best_gamma = 1
    best_eta = 0.6187849412845402

    xgb_cl = xgb.XGBClassifier(gamma=best_gamma,
                               max_depth=best_max_depth, eta=best_eta
                               )
    xgb_cl.fit(X_train, y_train)
    probs = xgb_cl.predict_proba(X_test)
    # normalize
    probs = probs / np.sum(probs, axis=0)

    # test accuracy

    actual = X_test.copy()
    actual['podium'] = y_test

    accuracyDF = actual[['season', 'round', 'podium']]
    places = ["first", "second", "third"]
    i = 1
    for place in places:
        globals()[place + "_place_probs"] = probs[:, i]
        i += 1

        accuracyDF[place + "_place_probs"] = globals()[place + "_place_probs"]
        actual[place + "_place_probs"] = globals()[place + "_place_probs"]
    third_place_probs
    accuracyDF['placement_prediction'] = 0

    def make_predictions(group):
        # Initialize the placement_prediction column
        group['placement_prediction'] = 0

        # Find the row(s) with the maximum first_place_probs and set placement_prediction to 1
        max_first_place_prob = group['first_place_probs'].max()
        first_pred_mask = group['first_place_probs'] == max_first_place_prob
        group.loc[first_pred_mask, 'placement_prediction'] = 1

        # Find the row(s) with the maximum second_place_probs among the remaining rows and set placement_prediction to 2
        remaining_group = group[~first_pred_mask]
        max_second_place_prob = remaining_group['second_place_probs'].max()
        second_pred_mask = group['second_place_probs'] == max_second_place_prob
        group.loc[second_pred_mask, 'placement_prediction'] = 2

        # Find the row(s) with the maximum third_place_probs among the remaining rows and set placement_prediction to 3
        remaining_group = remaining_group[~second_pred_mask]
        max_third_place_prob = remaining_group['third_place_probs'].max()
        third_pred_mask = group['third_place_probs'] == max_third_place_prob
        group.loc[third_pred_mask, 'placement_prediction'] = 3

        return group

    # Apply the make_predictions function to each group (round)
    accuracyDF = accuracyDF.groupby(['season', 'round']).apply(
        make_predictions).reset_index(drop=True)
    accuracyDF = accuracyDF.drop(
        ['first_place_probs', 'second_place_probs', 'third_place_probs'], axis=1)
    # saves df
    accuracyDF.to_csv("xgb_boost_predictions.csv")

    # find podium accuracy
    total_groups = 0
    nonzero_prediction_groups = 0
    grouped_df = accuracyDF.groupby(['season', 'round'])

    for group_key, group_data in grouped_df:
        total_groups += 1

        # Get the podium values and prediction placements for the group
        podium_values = group_data['podium'].values
        prediction_placements = group_data['placement_prediction'].values

        # Find the indices of the non-zero podium values
        non_zero_indices = np.nonzero(podium_values)[0]

        # Check if there are exactly 3 non-zero podium values and all corresponding prediction placements are non-zero
        if len(non_zero_indices) == 3 and (prediction_placements[non_zero_indices] != 0).all():
            nonzero_prediction_groups += 1

    podium_accuracy = nonzero_prediction_groups / total_groups
    print("Podium Accuracy (how often we go 3/3 in podium): " + str(podium_accuracy))

    # find first, second, third place accuracy
    per_round_first = actual.sort_values(
        by='first_place_probs', ascending=False).groupby(['season', 'round']).first()
    per_round_second = actual.sort_values(
        by='second_place_probs', ascending=False).groupby(['season', 'round']).first()
    per_round_third = actual.sort_values(
        by='third_place_probs', ascending=False).groupby(['season', 'round']).first()

    first_place_accuracy = sum(
        per_round_first['podium'] == 1)/len(per_round_first)
    second_place_accuracy = sum(
        per_round_second['podium'] == 2)/len(per_round_second)
    third_place_accuracy = sum(
        per_round_third['podium'] == 3)/len(per_round_third)

    print("First Place Accuracy: " + str(first_place_accuracy))
    print("Second Place Accuracy: " + str(second_place_accuracy))
    print("Third Place Accuracy: " + str(third_place_accuracy))

    # save model as xgb_classifier.joblib
    joblib.dump(xgb_cl, "xgb_classifier.joblib")

    # predict list of accuracies for first, second, third, podium
    return {"First Place Accuracy": first_place_accuracy, "Second Place Accuracy": second_place_accuracy, "Third Place Accuracy": third_place_accuracy, "Podium Accuracy": podium_accuracy}

# saves prediction df and prints out predictions


def run(predictions_csv_path, race_name="insert_race_name", pretrained_model_path="xgb_classifier.joblib"):
    data = pd.read_csv(predictions_csv_path)
    to_predict = data.drop(['driver', 'podium'], axis=1)
    xgb_cl = joblib.load(pretrained_model_path)

    cols_rearranged = ['season', 'round', 'weather_warm', 'weather_cold', 'weather_dry', 'weather_wet', 'weather_cloudy', 'grid', 'driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 'constructor_wins', 'constructor_standings_pos', 'driver_age', 'qualifying_secs', 'circuit_id_adelaide', 'circuit_id_albert_park', 'circuit_id_americas', 'circuit_id_bahrain', 'circuit_id_baku', 'circuit_id_brands_hatch', 'circuit_id_catalunya', 'circuit_id_detroit', 'circuit_id_estoril', 'circuit_id_galvez', 'circuit_id_hockenheimring', 'circuit_id_hungaroring', 'circuit_id_imola', 'circuit_id_indianapolis', 'circuit_id_interlagos', 'circuit_id_istanbul', 'circuit_id_jacarepagua', 'circuit_id_jerez', 'circuit_id_kyalami', 'circuit_id_magny_cours', 'circuit_id_marina_bay', 'circuit_id_monaco', 'circuit_id_monza', 'circuit_id_nurburgring', 'circuit_id_phoenix', 'circuit_id_red_bull_ring', 'circuit_id_ricard', 'circuit_id_rodriguez', 'circuit_id_sepang', 'circuit_id_shanghai', 'circuit_id_silverstone', 'circuit_id_sochi', 'circuit_id_spa', 'circuit_id_suzuka', 'circuit_id_valencia', 'circuit_id_villeneuve', 'circuit_id_yas_marina', 'circuit_id_yeongam', 'circuit_id_zandvoort', 'nationality_American', 'nationality_Australian', 'nationality_Austrian', 'nationality_Belgian', 'nationality_Brazilian',
                       'nationality_British', 'nationality_Canadian', 'nationality_Danish', 'nationality_Dutch', 'nationality_Finnish', 'nationality_French', 'nationality_German', 'nationality_Italian', 'nationality_Japanese', 'nationality_Mexican', 'nationality_Russian', 'nationality_Spanish', 'nationality_Swedish', 'constructor_alfa', 'constructor_arrows', 'constructor_bar', 'constructor_benetton', 'constructor_brabham', 'constructor_ferrari', 'constructor_footwork', 'constructor_force_india', 'constructor_haas', 'constructor_jaguar', 'constructor_jordan', 'constructor_larrousse', 'constructor_ligier', 'constructor_lotus_f1', 'constructor_mclaren', 'constructor_mercedes', 'constructor_minardi', 'constructor_prost', 'constructor_red_bull', 'constructor_renault', 'constructor_sauber', 'constructor_team_lotus', 'constructor_toro_rosso', 'constructor_toyota', 'constructor_tyrrell', 'constructor_williams', 'number_of_laps', 'circuit_length', 'circuit_id_miami', 'circuit_id_jeddah', 'circuit_id_qatar', 'circuit_id_long_beach', 'circuit_id_portugal', 'circuit_id_italy', 'circuit_id_buddh', 'circuit_id_fuji', 'circuit_id_pacific', 'circuit_id_europe', 'circuit_id_fair_park', 'circuit_id_dijon_prenois', 'circuit_id_rio', 'direction_a', 'direction_c', 'direction_nan', 'podium_SMA_nationality', 'podium_SMA_constructor', 'driver_avg_placement']
    to_predict = to_predict[cols_rearranged]

    # preds_race = xgb_cl.predict(to_predict)
    probs_preds_race_4 = xgb_cl.predict_proba(to_predict)
    # this normalizes probs, does not do anything useful at the moment
    probs = probs_preds_race_4 / np.sum(probs_preds_race_4, axis=0)

    predictionDF = data[['season', 'round', 'driver']]
    places = ["first", "second", "third"]
    i = 1
    for place in places:
        globals()[place + "_place_probs"] = probs[:, i]
        i += 1
        predictionDF[place + "_place_probs"] = globals()[place +
                                                         "_place_probs"]
    third_place_probs
    predictionDF['placement_prediction'] = 0

    # print(max_index_each_col)
    max_first_place_prob = predictionDF['first_place_probs'].max()
    first_pred_mask = predictionDF['first_place_probs'] == max_first_place_prob
    first_place_prediction = predictionDF.loc[predictionDF['first_place_probs'].idxmax(
    ), 'driver']
    predictionDF.loc[first_pred_mask, 'placement_prediction'] = 1

    # Find the row(s) with the maximum second_place_probs among the remaining rows and set placement_prediction to 2
    remaining_group = predictionDF[~first_pred_mask]
    max_second_place_prob = remaining_group['second_place_probs'].max()
    second_pred_mask = predictionDF['second_place_probs'] == max_second_place_prob
    second_place_prediction = predictionDF.loc[predictionDF['second_place_probs'].idxmax(
    ), 'driver']
    predictionDF.loc[second_pred_mask, 'placement_prediction'] = 2

    # Find the row(s) with the maximum third_place_probs among the remaining rows and set placement_prediction to 3
    remaining_group = remaining_group[~second_pred_mask]
    max_third_place_prob = remaining_group['third_place_probs'].max()
    third_pred_mask = predictionDF['third_place_probs'] == max_third_place_prob
    third_place_prediction = predictionDF.loc[predictionDF['third_place_probs'].idxmax(
    ), 'driver']
    predictionDF.loc[third_pred_mask, 'placement_prediction'] = 3

    print("First Place Prediction: " + first_place_prediction)
    print("Second Place Prediction: " + second_place_prediction)
    print("Third Place Prediction: " + third_place_prediction)

    # saves df
    predictionDF.to_csv("xgb_boost_predictions_for_" + race_name + ".csv")

    # saves df
    predictionDF.to_csv("xgb_boost_predictions.csv")
    # return list of predictions
    return {"First Pred": first_place_prediction, "Second Pred": second_place_prediction, "Third Pred": second_place_prediction}


# ONLY MODIFY BELOW THIS LINE
training_csv_path = "../data/after_miami.csv"
predictions_csv_path = "../../prediction_dfs/pred_df_race_monaco.csv"
race_name = "monaco"

# you do not have to retrain unless we updated the training data set(because we saved the weights of the most recent training under xgb_classifier.joblib")
train(training_csv_path)
run(predictions_csv_path, race_name)
