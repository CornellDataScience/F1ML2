from pairwise_ranking import NeuralNets
from pairwise_ranking import test_func
from pairwise_ranking import DataSets
import pandas as pd
import torch

def get_nn_results():
    """
    Get df containing results from running nn
    """

    df = {"Neural Net Precision": []}
    
    #d = DataSets("HOLY_after_race3.csv")
    d = DataSets("prediction_dfs/pred_df_race3.csv")
    
    #d = pd.read_csv("../prediction_dfs/pred_df_race3.csv")
    
    # dropping from sample data set (not necessary)
    #d.drop(columns=['podium_SMA_nationality'], inplace=True)
    #d.drop(columns=['podium_SMA_constructor'], inplace=True)

    model = NeuralNets(d.num_features)
    model.load_state_dict(torch.load("neural_nets.pth"))
    model.eval()

    for param in model.parameters():
        print(param.data)
    
    test_func(model(torch.tensor(d.testX.astype(float)).float()
                          ).detach().numpy(), d.test_df_with_driver)

    return df

if __name__ == '__main__':
    get_nn_results()