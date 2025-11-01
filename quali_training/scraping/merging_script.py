import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from requests import get

# full_df = pd.read_csv('../cleaning/FULL_F1_DF.csv')

in_frame = 'temp_race_miami.csv'  # replace with full csv
pred_frame = '../prediction_dfs/pred_df_race_miami.csv'  # replace w pred csv
# change link to data
in_url = 'https://www.formula1.com/en/results.html/2023/races/1208/miami/race-result.html'
out_name = "after_miami.csv"  # file to create


def race(in_csv, pred_csv, url, out_name, make_driver_avg=False):
    # webscraping
    full_df = pd.read_csv(in_csv)  # change dataframe
    response = get(url)
    pred_frame = pd.read_csv(pred_csv)

    html_soup = BeautifulSoup(response.text, 'html.parser')
    results = html_soup.find_all('td', class_='dark')
    pred_frame['podium'] = 0
    drivers = []
    i = 2
    while i <= 80:
        split = results[i].text.split("\n")
        drivers.append(split[1] + " " + split[2])
        i = i + 4
    name_to_id = {"Max Verstappen": "max_verstappen", "Sergio Perez": "perez", "Fernando Alonso": "alonso", "Carlos Sainz": "sainz", "Lewis Hamilton": "hamilton", "Lance Stroll": "stroll", "George Russell": "russell", "Valtteri Bottas": "bottas", "Pierre Gasly": "gasly", "Alexander Albon": "albon",
                  "Yuki Tsunoda": "tsunoda", "Logan Sargeant": "sargeant", "Kevin Magnussen": "kevin_magnussen", "Nyck De Vries": "de_vries", "Nico Hulkenberg": "hulkenberg", "Zhou Guanyu": "zhou", "Lando Norris": "norris", "Esteban Ocon": "ocon", "Charles Leclerc": "leclerc", "Oscar Piastri": "piastri"}
    driver_ids = []
    for driver in drivers:
        driver_ids.append(name_to_id.get(driver))

    for i in range(len(driver_ids)):
        pred_frame.loc[pred_frame['driver'] == driver_ids[i], 'podium'] = i+1
    full_df = full_df.append(pred_frame)
    if 'Unnamed: 0' in full_df.columns:
        full_df = full_df.drop(columns=['Unnamed: 0'])
    if 'qualifying_time' in full_df.columns:
        full_df = full_df.drop(columns=['qualifying_time'])
    if 'index' in full_df.columns:
        full_df = full_df.drop(columns=['index'])
    if 'Unnamed: 0.1' in full_df.columns:
        full_df = full_df.drop(columns=['Unnamed: 0.1'])
    if 'avg' in full_df.columns:
        full_df = full_df.drop(columns=['avg'])

    """
    # adding transformations from feature engineering
    # order by season
    full_df['timestamp'] = full_df['season'] + full_df['round'].div(100)
    full_df = full_df.sort_values('timestamp')
    # Undo One-Hot Encode for Nationality
    cons = [n for n in full_df.columns if n.startswith('constructor')]
    cons.remove('constructor_points')
    cons.remove('constructor_wins')
    cons.remove('constructor_standings_pos')
    full_df_cons = full_df[cons]
    full_df['constructor'] = full_df_cons.idxmax(1)
    full_df_final = full_df[0:0]
    full_df_final['podium_SMA_constructor'] = None
    full_df_final.head()
    # 68,71,72,82,84,88,90,92,93
    # Creating Moving Averages
    full_df_temp = full_df[0:0]
    full_df_temp['podium_SMA_constructor'] = None
    final_cols = list(full_df_final.columns)
    temp_cols = list(full_df_temp.columns)
    # for i in range(0,len(final_cols)):
    #    print(final_cols[i] == temp_cols[i])
    for constructor in cons:
        full_df_temp = full_df.loc[full_df[constructor] == 1]
        full_df_temp['podium_SMA_constructor'] = full_df_temp['podium'].rolling(30).mean()
        full_df_temp = full_df_temp.fillna(full_df_temp['podium'].head(30).mean())

        full_df_temp = full_df.loc[full_df[constructor] == 1]
        full_df_temp['podium_SMA_constructor'] = full_df_temp['podium'].rolling(30).mean()
        full_df_temp = full_df_temp.fillna(full_df_temp['podium'].head(30).mean())
        full_df_final = pd.concat([full_df_final, full_df_temp], axis=0)
    full_df_final = full_df_final.sort_values('timestamp')
    full_df_final.to_csv("test_merging_script.csv")
    """
    # CHANGE FILE NAME

    if make_driver_avg:
        if 'driver_avg_placement' in full_df.columns:
            full_df = full_df.drop(columns=['driver_avg_placement'])
        driver_groups = full_df.groupby(['driver'])
        driver_groups

        driver_placement_avg = pd.DataFrame(
            columns=['driver', 'season', 'round', 'avg'])
        driver_placement_avg

        for name, group in driver_groups:
            for i in range(1, len(group)+1):
                average = group.rolling(i).mean().podium.iloc[i-1]
                driver_placement_avg = driver_placement_avg.append({'driver': group.driver.iloc[i-1],
                                                                    'round': group['round'].iloc[i-1],
                                                                    'season': group.season.iloc[i-1],
                                                                    'avg': average}, ignore_index=True)

        merged_df = full_df.merge(driver_placement_avg, on=[
                                  'driver', 'season', 'round'])
        driver_groups = merged_df.groupby('driver')
        driver_avg_placement = driver_groups.avg.shift(1)
        merged_df['driver_avg_placement'] = driver_avg_placement
        merged_df
        full_df = merged_df
        full_df['driver_avg_placement'].fillna(15, inplace=True)
        full_df.drop(columns=['avg'], inplace=True)

    full_df.to_csv(out_name)


race(in_frame, pred_frame, in_url, out_name, make_driver_avg=True)
