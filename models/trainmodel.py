import xgboost as xgb
import numpy as np
import pandas as pd

# Import & Split Train/Test Data
df = pd.read_csv('../data/HOLYv1.csv').drop(columns=['Unnamed: 0'])
id = 0
for s in set(df['season']):
    for r in set(df['round']):

        df.loc[(df['season'] == s) & (df['round'] == r), 'id'] = id
        id += 1
df.sort_values('id', inplace=True)

def process_df(df):
    y = df.loc[:, 'podium']

    return df.drop(columns=['podium', 'driver', 'season', 'round', 'id']), y

# Create Tuned Model
model = xgb.XGBRanker(  **{'n_estimators': 169,
  'max_depth': 7,
  'learning_rate': 0.023674177607202307,
  'colsample_bytree': 0.5438701967084194,
  'subsample': 0.4532488227992929,
  'alpha': 0.10664514806429565,
  'lambda': 0.09342144391433568,
  'min_child_weight': 140.49671900290397,
  
  "verbosity": 0,  # 0 (silent) - 3 (debug)
  "objective": "rank:pairwise",}
)

x, y = process_df(df)
groups = df.groupby('id').size().to_frame('size')['size'].to_numpy()
model.fit(x, y, group=groups, verbose=True)
model.save_model("xgbranker_model.json")
