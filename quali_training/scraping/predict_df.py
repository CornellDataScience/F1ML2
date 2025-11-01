import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from requests import get

import warnings
warnings.filterwarnings("ignore")

drop_cols = ['Unnamed: 0', 'Unnamed: 0.1']  # COLUMNS TO DROP
weather_warm = True  # JUST GOOGLE THE WEATHER FOR RACE DAY AND PUT IN HERE
weather_cold = False
weather_dry = False
weather_wet = True
weather_cloudy = True
# KEEP CIRCUIT_ID, CHANGE TO RACE TRACK ID. LOOK AT COLUMN NAMES IN DF TO FIND WHAT MATCHES
location_id = "circuit_id_monaco"
curr_round = 6  # 6 = MONACO
curr_season = 2023  # CHANGE ON NEW SEASON

# direction_a: anticlockwise, direction_c:clockwise, direction_nan: figure 8
direction = 'direction_c'

# CHANGE TO CURRENT FULL DF IN USE
full_df_path = 'scraping/after_miami.csv'


def get_sec(time_str):
    """Get seconds from time."""
    m, s = time_str.split(':')
    return int(m) * 60 + float(s)


# CHANGE TO DESIRED OUTPUT CSV FILE NAME
def make_pred_df(build_csv=True, out_name="pred_df_race_monaco.csv", full_df_path=full_df_path):
    # ONLY CHANGE IF WE MAKE A NEW PRED DF TEMPLATE
    full_df = pd.read_csv(full_df_path)
    for col in drop_cols:
        if col in full_df.columns:
            full_df.drop(columns=[col], inplace=True)

    race1_df = pd.read_csv('2023_PRED_TEMPLATE.csv')  # run from root
    for col in drop_cols:
        if col in race1_df.columns:
            race1_df.drop(columns=[col], inplace=True)

    for col in race1_df.columns:
        if 'circuit_id' in col:
            race1_df[col] = 0
    race1_df[location_id] = 1

    for col in race1_df.columns:
        if 'direction_' in col:
            race1_df[col] = 0
    race1_df[direction] = 1

    race1_df['season'] = curr_season
    race1_df['round'] = curr_round

    url = 'https://www.formula1.com/en/results.html/2023/races/1210/monaco/starting-grid.html'
    # CHANGE THIS TO THE LINK WITH QUALIFYING RESULTS. NEED TO RE-WRITE WHOLE SCRIPT IF FORMAT CHANGES IN 2024

    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    results = html_soup.find_all('td', class_='dark')
    race1_df['weather_warm'] = weather_warm
    race1_df['weather_cold'] = weather_cold
    race1_df['weather_dry'] = weather_dry
    race1_df['weather_wet'] = weather_wet
    race1_df['weather_cloudy'] = weather_cloudy

    race1_df['number_of_laps'] = 78
    race1_df['circuit_length'] = 3.337

    race1_df['podium'] = 0
    drivers = []
    grid = []
    times = []

    g = 0
    i = 2

    while i <= 80:  # CHANGE THIS TO HIGHEST WORKING NUMBER
        split = results[i].text.split("\n")
        splitg = results[g].text.split("\n")
        drivers.append(split[1] + " " + split[2])
        grid.append(splitg[0])
        g = g + 4
        i = i + 4

    for res in results:
        txt = res.text
        if ':' in txt or txt == '':
            times.append(txt)
    new_times = []
    for j, time in enumerate(times):
        if time != '':
            new_times.append(get_sec(time))
        elif time == '':
            # if j > 0 and j < len(times) - 1:
            new_times.append(new_times[-1] + 0.01)
    new_times = np.array(new_times)
    new_times = new_times - min(new_times)
    name_to_id = {"Max Verstappen": "max_verstappen", "Sergio Perez": "perez", "Fernando Alonso": "alonso", "Carlos Sainz": "sainz", "Lewis Hamilton": "hamilton", "Lance Stroll": "stroll", "George Russell": "russell", "Valtteri Bottas": "bottas", "Pierre Gasly": "gasly", "Alexander Albon": "albon",
                  "Yuki Tsunoda": "tsunoda", "Logan Sargeant": "sargeant", "Kevin Magnussen": "kevin_magnussen", "Nyck De Vries": "de_vries", "Nico Hulkenberg": "hulkenberg", "Zhou Guanyu": "zhou", "Lando Norris": "norris", "Esteban Ocon": "ocon", "Charles Leclerc": "leclerc", "Oscar Piastri": "piastri"}
    # CHANGE THIS IN 2024. MAPS NAMES FOUND IN HTML TO DRIVER_IDS IN DF

    driver_ids = []
    for driver in drivers:
        driver_ids.append(name_to_id.get(driver))

    name_to_grid = {}
    for x in range(0, len(grid)):
        name_to_grid.update({driver_ids[x]: grid[x]})

    name_to_time = {}
    for x in range(len(times)):
        name_to_time.update({driver_ids[x]: new_times[x]})

    for i in range(len(driver_ids)):
        race1_df.loc[race1_df['driver'] == driver_ids[i], 'grid'] = grid[i]
        race1_df.loc[race1_df['driver'] ==
                     driver_ids[i], 'qualifying_secs'] = new_times[i]

    # RECOMPUTE DRIVER AVG
    full_df = full_df.append(race1_df)
    if True:
        if 'driver_avg_placement' in full_df.columns:
            full_df = full_df.drop(columns=['driver_avg_placement'])
        driver_groups = full_df.groupby(['driver'])

        driver_placement_avg = pd.DataFrame(
            columns=['driver', 'season', 'round', 'avg'])

        for name, group in driver_groups:
            for i in range(1, len(group)+1):
                average = group.rolling(i).mean().podium.iloc[i-1]
                driver_placement_avg = pd.concat([driver_placement_avg, pd.DataFrame.from_records({'driver': [group.driver.iloc[i-1]],
                                                                                                   'round': [group['round'].iloc[i-1]],
                                                                                                   'season': [group.season.iloc[i-1]],
                                                                                                   'avg': [average]})], ignore_index=False)

        merged_df = full_df.merge(driver_placement_avg, on=[
                                  'driver', 'season', 'round'])
        driver_groups = merged_df.groupby('driver')
        driver_avg_placement = driver_groups.avg.shift(1)
        merged_df['driver_avg_placement'] = driver_avg_placement
        full_df = merged_df
        full_df['driver_avg_placement'].fillna(15, inplace=True)
        full_df.drop(columns=['avg'], inplace=True)
        print(full_df)

        # RECOMPUTE SMA FEATS

        # SCRIPT
        # Order by season and round
        # full_df['timestamp'] = full_df['season'] + \
        #     full_df['round'].div(100)
        # full_df = full_df.sort_values('timestamp')

        # # Undo OneHot Encoding for Nationality
        # nats = [n for n in full_df.columns if n.startswith('nationality')]
        # df_nats = full_df[nats]
        # full_df['nationality'] = df_nats.idxmax(1)

        # # Create Moving Averages
        # df_final = full_df[0:0]
        # df_final['podium_SMA_nationality'] = None
        # df_final.head()

        # df_temp = full_df[0:0]
        # df_temp['podium_SMA_nationality'] = None

        # final_cols = list(df_final.columns)
        # temp_cols = list(df_temp.columns)

        # for nationality in nats:
        #     df_temp = full_df.loc[full_df[nationality] == 1]
        #     df_temp['podium_SMA_nationality'] = df_temp['podium'].rolling(
        #         30).mean()
        #     df_temp = df_temp.fillna(df_temp['podium'].head(30).mean())

        #     df_temp = full_df.loc[full_df[nationality] == 1]
        #     df_temp['podium_SMA_nationality'] = df_temp['podium'].rolling(
        #         30).mean()
        #     df_temp = df_temp.fillna(df_temp['podium'].head(30).mean())
        #     df_final = pd.concat([df_final, df_temp], axis=0)
        # df_final.drop(columns=['timestamp', 'nationality'], inplace=True)

        # new_race_df = df_final.loc[(df_final.season == curr_season)
        #                            & (df_final['round'] == curr_round)]

        new_race_df = full_df.loc[(full_df.season == curr_season)
                                  & (full_df['round'] == curr_round)]
        print(new_race_df)

        if build_csv:
            new_race_df.to_csv(out_name)


make_pred_df(build_csv=True, full_df_path=full_df_path)
