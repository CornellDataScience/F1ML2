import pandas as pd
import xgboost as xgb
import numpy as np

# Load Model Here
model = xgb.XGBRanker()
model.load_model("xgbranker_model.json")

# Initial DF Processing
df = pd.read_csv('../data/HOLYv1.csv').drop(columns=['Unnamed: 0'])
id = 0
for s in set(df['season']):
    for r in set(df['round']):
        df.loc[(df['season'] == s) & (df['round'] == r), 'id'] = id
        id += 1
df.sort_values('id', inplace=True)


p_df = pd.read_csv('../../pred_df_race_monaco.csv')
p_df.reindex(columns=df.columns)
p_df['pred'] = model.predict(p_df.reindex(columns=df.columns).drop(
    columns=['podium', 'driver', 'season', 'round', 'id']))
p_df.sort_values('pred', inplace=True)

names = p_df[['driver', 'pred']]

for index, row in p_df.iterrows():
    print(row['driver'] + ' : ' + str(row['pred']))
