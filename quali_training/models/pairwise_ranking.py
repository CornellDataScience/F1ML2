import torch
from torch import nn

from torch.nn import MarginRankingLoss

from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from random import sample
from numpy.random import seed
torch.manual_seed(42)


class PairwiseRanking(nn.Module):
    '''
    Simple logistic regressor
    '''

    def __init__(self, num_features) -> None:
        super().__init__()

        self.linear = nn.Linear(num_features, 10)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        res = self.linear(x)
        res = self.sigmoid(res)

        return res


class NeuralNets(nn.Module):
    '''
    Neural net
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        res = self.linear_relu_stack(x)
        # res = self.sigmoid(res)

        return res


class DataSets():

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file)
        # columns to drop for FULL_F1_DF.csv
        # df.drop(columns=['driver','driver_avg_placement'], inplace=True)

        # columns to drop for HOLYv1.csv
        # if('podium_SMA_constructor' in df.columns):
        #    df.drop(columns=['podium_SMA_constructor'], inplace=True)

        self.test_df_with_driver = df[df['season']
                                      > 2018].reset_index(drop=True)
        if ('driver' in df.columns):
            df.drop(columns=['driver'], inplace=True)

        if ('Unnamed: 0' in df.columns):
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'podium_SMA_constructor' in df.columns:
            df.drop(columns=['podium_SMA_constructor'], inplace=True)
        if 'podium_SMA_nationality' in df.columns:
            df.drop(columns=['podium_SMA_nationality'], inplace=True)
        # df.dropna(inplace=True)
        df = df.drop(
            columns=df.columns[df.columns.str.startswith('circuit_id')])
        df = df.drop(
            columns=df.columns[df.columns.str.startswith('nationality')])
        # df.drop(columns=df.columns[df.columns.str.startswith('constructor_')], inplace=True)
        self.train_df = df[df["season"] < 2014].reset_index(drop=True)
        self.val_df = df[(df['season'] > 2014) & (
            df['season'] <= 2018)].reset_index(drop=True)
        self.test_df = df[df['season'] > 2018].reset_index(drop=True)
        self.trainX = self.train_df.drop(columns=['podium'])
        # print(df.columns[df.isna().any()].tolist())
        # for column in self.trainX.columns:
        #    print(column)

        self.train_scaler = StandardScaler()

        self.train_scaler.fit(self.trainX)

        self.trainX = self.train_scaler.transform(self.trainX)

        self.trainy = self.train_df['podium']

        self.valX = self.val_df.drop(columns=['podium'])
        self.valy = self.val_df['podium']

        self.valX = self.train_scaler.transform(self.valX)

        self.testX = self.test_df.drop(columns=['podium'])
        self.testy = self.test_df['podium']

        self.testX = self.train_scaler.transform(self.testX)

        # print(self.trainX)

        self.num_features = self.trainX.shape[1]


def loss_func(preds, y, df, lmbda=0, margin=5, max_place=6):
    '''
    Implements pairwise ranking function. 

    Args:
        preds - shape (n, ) of predictions for all the races 
        y - shape (n, )  of true podium placements 

        df - dataframe of everything, in same order
        margin - float of how much to incentivize margin by

    Returns:
        loss - float of loss func
    '''

    loss = 0
    num = 0
    for round, season in set([(x, y) for x, y in df[['round', 'season']].values.tolist()]):
        race_df = df[(df['round'] == round) & (df['season'] == season)]

        pairwise = list(itertools.combinations(range(len(race_df)), 2))[:190]

        for i, j in pairwise:

            podium_i = race_df.iloc[i]
            podium_j = race_df.iloc[j]

            if podium_i['podium'] > podium_j['podium']:
                temp = podium_j
                podium_i = podium_j
                # podium_j = temp
            if min(i, j) < max_place:
                loss += max(preds[podium_j.name] +
                            margin - preds[podium_i.name], 0)

            num += 1

            # print(loss)
    return loss / num


def test_func(preds, df):
    '''
    Returns (% correct for first place, % correct for podium)
    '''
    num_correct_top_one = 0
    num_total_indiv = 0
    num_correct_second = 0
    num_correct_third = 0
    num_correct_top_two = 0
    num_total_indiv = 0
    num_correct_second = 0
    num_correct_third = 0
    num_correct_top_two = 0
    num_correct_podium = 0
    num_total_podium = 0
    num_correct_podium_order = 0
    num_correct_podium_order = 0

    df['pred'] = preds

    for round, season in set([(x, y) for x, y in df[['round', 'season']].values.tolist()]):
        race_df = df[(df['round'] == round) & (df['season'] == season)]

        top_pred_id = race_df.sort_values(
            by=['pred'], ascending=False).iloc[:3]
        real_pod_id = race_df.sort_values(
            by=['podium'], ascending=True).iloc[:3]

        num_correct_top_one += top_pred_id.iloc[0].name == real_pod_id.iloc[0].name
        num_total_indiv += 1

        # num_correct_second += top_pred_id.iloc[1].name == real_pod_id.iloc[1].name
        num_correct_third += top_pred_id.iloc[2].name == real_pod_id.iloc[2].name
        num_correct_top_two += (top_pred_id.iloc[0].name == real_pod_id.iloc[0].name) and (
            top_pred_id.iloc[1].name == real_pod_id.iloc[1].name)
        num_correct_podium_order += (top_pred_id.iloc[0].name == real_pod_id.iloc[0].name) and (
            top_pred_id.iloc[1].name == real_pod_id.iloc[1].name) and (top_pred_id.iloc[2].name == real_pod_id.iloc[2].name)
        # num_correct_second += top_pred_id == real_pod_id
        # num_correct_second += 1

        top_3 = set(race_df.sort_values(
            by=['pred'], ascending=False).iloc[:3].index)

        # prints out the names of drivers
        # print(f"{round} and {season}")
        """for i in range(6):
            driver = race_df.sort_values(
                by=['pred'], ascending=False).iloc[i].driver
            print(driver)
        print('NEXT RACE')"""

        real_3 = set(race_df.sort_values(
            by=['podium'], ascending=True).iloc[:3].index)

        # len(top_3.intersection(real_3))
        num_correct_podium += top_3 == real_3
        num_total_podium += 1

    return {'first only': num_correct_top_one/num_total_indiv,
            'second only': num_correct_second/num_total_indiv,
            'third only': num_correct_third/num_total_indiv,
            'top two in order': num_correct_top_two/num_total_indiv,
            'podium in order': num_correct_podium_order / num_total_indiv,
            'top three any order': num_correct_podium/num_total_podium}


if __name__ == '__main__':

    d = DataSets('HOLY_after_race3.csv')
    # d = DataSets('scraping/temp_race_miami.csv')
    # print("holy", d.test_df_with_driver.columns)
    epochs = 100

    model_exists = "neural_nets.pth"
    if True and model_exists:  # TRAIN if False
        model = NeuralNets(d.num_features)
        model.load_state_dict(torch.load(model_exists))
        # model = torch.load(model_exists)
    else:
        model = NeuralNets(d.num_features)
        # print(d.num_features)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

        losses = []
        val_losses = []

        for i in tqdm(range(epochs)):

            # Backpropagation
            optimizer.zero_grad()
            # Compute prediction error
            pred = model(torch.tensor(d.trainX.astype(float)).float())

            loss = loss_func(pred, d.trainy, d.train_df)

            val_pred = model(torch.tensor(d.valX.astype(float)).float())

            # calculate avg of previous num_prev val_losses (after at least num_prev epochs)
            num_prev = 8
            avg = 0
            if i > num_prev:
                for j in range(len(val_losses)-num_prev, len(val_losses)):
                    avg += val_losses[j]

                avg /= num_prev

                val_loss = loss_func(val_pred, d.valy, d.val_df)
                if (val_loss.detach().numpy() > avg):
                    break

            val_loss = loss_func(val_pred, d.valy, d.val_df)

            tqdm.write(f"training loss: {loss}")
            tqdm.write(f"val loss: {val_loss}")

            # print(model.linear.weight)

            loss.backward()
            optimizer.step()

            val_pred = model(torch.tensor(
                d.valX.astype(float)).float())

            loss = loss.detach().item()

            with torch.no_grad():
                val_loss = loss_func(
                    val_pred, d.valy, d.val_df).detach().item()

            tqdm.write(f"training loss: {loss}")
            tqdm.write(f"val loss: {val_loss}")

            val_losses.append(val_loss)
            losses.append(loss)

        # torch.save(model, 'neural_nets.pth')
        torch.save(model.state_dict(), 'neural_nets.pth')
        print("SAVED")

        plt.title("Training loss")

        plt.plot(list(range(1, len(losses) + 1)), losses, label='training')

        plt.plot(list(range(1, len(val_losses) + 1)),
                 val_losses, label='validation')

        plt.legend()
        plt.savefig('loss.png')

    print(test_func(model(torch.tensor(d.testX.astype(float)).float()
                          ).detach().numpy(), d.test_df_with_driver))

    want_predict = True
    if want_predict:
        test_df = pd.read_csv("prediction_dfs/pred_df_race_monaco.csv")
        if 'podium_SMA_constructor' in test_df.columns:
            test_df.drop(columns='podium_SMA_constructor', inplace=True)
        if 'podium_SMA_nationality' in test_df.columns:
            test_df.drop(columns='podium_SMA_nationality', inplace=True)
        test_df_with_driver = test_df.copy()
        test_df.drop(['podium', 'driver'], axis=1, inplace=True)
        if 'Unnamed: 0' in test_df.columns:
            test_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        test_df = test_df.drop(
            columns=test_df.columns[test_df.columns.str.startswith('circuit_id')])
        test_df = test_df.drop(
            columns=test_df.columns[test_df.columns.str.startswith('nationality')])
        # print(test_df.columns)
        # for column in test_df.columns:
        #    print(column)

        test_df = d.train_scaler.transform(test_df)
        # print("columns", len(test_df))
        pred = model(torch.tensor(test_df.astype(float)).float())
        pred = pred.tolist()
        # print(test_df_with_driver['driver'])
        out_df = pd.DataFrame(columns=['driver', 'score'])
        out_df['driver'] = test_df_with_driver['driver']
        out_df['score'] = pred
        # print(out_df)
        out_df = out_df.sort_values(by=['score'], ascending=False)
        print(out_df)
        # print(test_func(model(torch.tensor(d.testX.astype(float)).float()
        #                 ).detach().numpy(), test_df_with_driver))
        # print("hello")
