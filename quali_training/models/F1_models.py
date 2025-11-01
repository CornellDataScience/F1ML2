# import libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, LassoLarsCV
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

np.set_printoptions(precision=4)

class F1_regressions:
    """
    Attributes:
    SKLearn model objects:
    linear_model
    svm_model
    self.nn
    Regression model scores:
    linear_score
    svm_score
    nn_score
    data_file: csv to use
    test_year: year for testing
    run? (bools)
    linear
    svm
    nn
    preds: prediction results
    comparison_dict: compare models
    Methods: 
    run_all_models: run all models and get results
    """

    def __init__(self, data_file='../data/HOLYv1.csv', test_year=2022, linear=True, lasso=False,
    lassocv=False, lassolarscv=False, svm=False, forest=False, logit=False, nn=False):
        """
        Creates F1_regressions object
        Optional params: 
        data_file: source file
        test_year: year for test data
        linear: Run linear models
        svm: Run svm models
        nn: Run nn models
        """
        self.linear_model = None
        self.svm_model = None
        self.forest_model = None
        self.logit_model = None
        self.nn = None
        self.lasso_model = None
        self.lassocv_model = None
        self.lassolarscv = None

        self.linear_scaler = None

        self.linear_score = 0
        self.svm_score = 0
        self.forest_score = 0
        self.logit_score = 0
        self.nn_score = 0
        self.lasso_score = 0
        self.lassocv_score = 0
        self.lassolarscv_score = 0

        self.data_file = data_file
        self.test_year = test_year
        self.linear = linear
        self.svm = svm
        self.forest = forest
        self.logit = logit
        self.nn = nn
        self.lasso = lasso
        self.lassocv = lassocv
        self.lassolarscv = lassolarscv

        self.lassocv_features = []
        self.lassolarscv_features = []
        self.lassocv_coefs = []
        self.lassolarscv_coefs = []
        self.zeroed_coefs = ""
        self.non_zeroed_coefs = ""

        self.preds = None

        self.comparison_dict = {'model': [],
                                'params': [],
                                'score': []}

    def get_preds(self):
        return self.preds

    def get_comparison_dict(self):
        return self.comparison_dict
    
    def get_lassocv_coefs(self):
        return self.lassocv_coefs
    
    def get_lassocv_features(self):
        return self.lassocv_features
    
    def get_lassolarscv_coefs(self):
        return self.lassolarscv_coefs
    
    def get_lassolarscv_features(self):
        return self.lassolarscv_features

    def get_zeroed_lassocv_coefs(self):
        return self.zeroed_coefs
    
    def get_non_zeroed_lassocv_coefs(self):
        return self.non_zeroed_coefs

    def score_regression(self, model, df, scaler, test_year=2022):
        """ Scores a sklearn regression model
        model: the sklearn model to be scored
        df: original results df
        scaler: standard scaler object
        test_year: year used as test set
        Returns model precision score, prediction results"""

        score = 0
        preds = pd.DataFrame()
        for circuit in df[df.season == test_year]['round'].unique():
            test = df[(df.season == test_year) & (df['round'] == circuit)]
            X_test = test.drop(['driver', 'podium'], axis=1)
            y_test = test.podium

            # scaling
            X_test = pd.DataFrame(scaler.transform(
                X_test), columns=X_test.columns)

            # make predictions
            prediction_df = pd.DataFrame(
                model.predict(X_test), columns=['results'])
            prediction_df['podium'] = y_test.reset_index(drop=True)
            prediction_df['actual'] = prediction_df.podium.map(
                lambda x: 1 if x == 1 else 0)
            prediction_df.sort_values('results', ascending=True, inplace=True)
            prediction_df.reset_index(inplace=True, drop=True)
            prediction_df['predicted'] = prediction_df.index
            prediction_df['predicted'] = prediction_df.predicted.map(
                lambda x: 1 if x == 0 else 0)
            prediction_df['round'] = prediction_df.predicted.map(
                lambda x: circuit)
            preds = pd.concat([prediction_df])

            score += precision_score(prediction_df.actual,
                                     prediction_df.predicted)

        model_score = score / len(df[df.season == test_year]['round'].unique())
        print("model score: ", model_score, "  ", score)
        return model_score, preds

    def train_linear_regression(self, df, X_train, y_train, scaler, test_year=2022):
        """
        Train linear model and make test predictions
        df: df to use
        X_train, y_train: train set
        scaler: standard scaler object
        test_year: year for testing
        """
        model = LinearRegression(fit_intercept=True)  # can change to false
        model.fit(X_train, y_train)

        scored_regression = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score = scored_regression[0]
        self.linear_score = model_score

        preds = scored_regression[1]

        self.comparison_dict['model'].append('linear_regression')
        self.comparison_dict['params'].append(
            "Fit Intercept = True")  # corresponds with above
        self.comparison_dict['score'].append(model_score)

        comparison_df = pd.DataFrame(self.comparison_dict)
        # print(comparison_df)
        return model, preds

    def train_lasso_regression(self, df, X_train, y_train, scaler, test_year=2022, alpha = 100.0):
        """
        Train Lasso model and make test predictions
        df: df to use
        X_train, y_train: train set
        scaler: standard scaler object
        test_year: year for testing
        alpha: the alpha used in model training
        """
        model = Lasso(fit_intercept='True', alpha = alpha)  # can change to false
        model.fit(X_train, y_train)

        scored_regression = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score = scored_regression[0]
        self.lasso_score = model_score

        preds = scored_regression[1]

        self.comparison_dict['model'].append('lasso_regression')
        self.comparison_dict['params'].append(
            "Fit Intercept = True")  # corresponds with above
        self.comparison_dict['score'].append(model_score)

        comparison_df = pd.DataFrame(self.comparison_dict)
        # print(comparison_df)
        return model, preds

    def train_lassocv_regression(self, df, X_train, y_train, scaler, test_year=2000):
        """
        Train LassoCV model and make test predictions
        df: df to use
        X_train, y_train: train set
        scaler: standard scaler object
        test_year: year for testing
        n_alphas: 
        """
        model = LassoCV(normalize = True)  # can change to false
        model.fit(X_train, y_train)
        feature_names = list(X_train.columns.values)
        self.lassocv_features = feature_names
        self.lassocv_coefs = model.coef_

        scored_regression = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score = scored_regression[0]
        self.lassocv_score = model_score    

        preds = scored_regression[1]

        self.comparison_dict['model'].append('lassocv_regression')
        self.comparison_dict['params'].append(
            "Fit Intercept = True")  # corresponds with above
        self.comparison_dict['score'].append(model_score)

        comparison_df = pd.DataFrame(self.comparison_dict)
        # print(comparison_df)
        return model, preds

    def train_lassolarscv_regression(self, df, X_train, y_train, scaler, test_year=2022):
        """
        Train LassoLarsCV model and make test predictions
        df: df to use
        X_train, y_train: train set
        scaler: standard scaler object
        test_year: year for testing
        """
        model = LassoLarsCV(cv = 10)
        model.fit(X_train, y_train)
        feature_names = list(X_train.columns.values)
        self.lassolarscv_features = feature_names
        self.lassolarscv_coefs = model.coef_

        scored_regression = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score = scored_regression[0]
        self.lassolarscv_score = model_score    

        preds = scored_regression[1]

        self.comparison_dict['model'].append('lassolarscv_regression')
        self.comparison_dict['params'].append(
            "Fit Intercept = True")  # corresponds with above
        self.comparison_dict['score'].append(model_score)

        comparison_df = pd.DataFrame(self.comparison_dict)
        # print(comparison_df)
        return model, preds


    def train_svm(self, df, X_train, y_train, scaler, test_year=2022):
        model = svm.SVR(gamma=.01, C=1, kernel='linear')
        model.fit(X_train, y_train)

        score_svm = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score_svm = score_svm[0]
        self.svm_score = model_score_svm
        preds_svm = score_svm[1]

        self.comparison_dict['model'].append('svm_regressor')
        self.comparison_dict['params'].append(
            "gamma = .01, C = 1, kernel = 'linear'")
        self.comparison_dict['score'].append(model_score_svm)
        return model, preds_svm

    def train_svm(self, df, X_train, y_train, scaler, test_year=2022):
        model = svm.SVR(gamma=.01, C=1, kernel='linear')
        model.fit(X_train, y_train)

        score_svm = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score_svm = score_svm[0]
        self.svm_score = model_score_svm
        preds_svm = score_svm[1]

        self.comparison_dict['model'].append('svm_regressor')
        self.comparison_dict['params'].append(
            "gamma = .01, C = 1, kernel = 'linear'")
        self.comparison_dict['score'].append(model_score_svm)
        return model, 

    def train_forest(self, df, X_train, y_train, scaler, test_year = 2022):
        model = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", min_samples_leaf = 3, random_state = 10)
        model.fit(X_train, y_train)

        score_forest = self.score_regression(model, df, scaler, test_year=test_year)
        model_score_forest = score_forest[0]
        self.forest_score = model_score_forest
        preds_forest = score_forest[1]

        self.comparison_dict['model'].append("Random_Forest")
        self.comparison_dict['params'].append("n_e=100, sqe, min_s_leaf=3,state=10")
        self.comparison_dict['score'].append(model_score_forest)

        return model, preds_forest

    def train_logistic_regression(self, df, X_train, y_train, scaler, test_year=2022):
        """
        Train linear model and make test predictions
        df: df to use
        X_train, y_train: train set
        scaler: standard scaler object
        test_year: year for testing
        """
        model = LogisticRegression()  # can change to false
        model.fit(X_train, y_train)

        scored_regression = self.score_regression(
            model, df, scaler, test_year=test_year)
        model_score = scored_regression[0]
        self.logit_score = model_score

        preds = scored_regression[1]

        self.comparison_dict['model'].append('logit_regression')
        self.comparison_dict['params'].append(
            "Fit Intercept = True")  # corresponds with above
        self.comparison_dict['score'].append(model_score)

        comparison_df = pd.DataFrame(self.comparison_dict)
        # print(comparison_df)
        return model, preds

    def score_classification(self, model, df, scaler, test_year=2022):
        # score = 0
        # for circuit in df[df.season == 2022]['round'].unique():

        #     test = df[(df.season == 2022) & (df['round'] == circuit)]
        #     X_test = test.drop(['driver', 'podium'], axis = 1)
        #     y_test = test.podium

        #     #scaling
        #     X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        #     # make predictions
        #     prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        #     prediction_df['actual'] = y_test.reset_index(drop = True)
        #     prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        #     prediction_df.reset_index(inplace = True, drop = True)
        #     prediction_df['predicted'] = prediction_df.index
        #     prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        #     score += precision_score(prediction_df.actual, prediction_df.predicted)

        # model_score = score / len(df[df.season == 2022]['round'].unique())
        # return model_score, prediction_df
        pass

    def train_nn(self, df, X_train, y_train, scaler, test_year=2022):
        # model = MLPClassifier(hidden_layer_sizes = (75,25,50,10),
        #                       activation = 'identity', solver = 'lbfgs', alpha = 0.01623776739188721, random_state = 1)

        # # NN
        # model.fit(X_train, y_train)

        # score_nn = score_classification(model)
        # model_score_nn = score_nn[0]
        # preds_nn = score_nn[1]
        pass

    def run_all_models(self, test_year=2022):
        data = pd.read_csv(self.data_file)

        df = data.copy()
        if 'qualifying_time' in df.columns:
            df = df.drop(['qualifying_time'], axis=1)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)
        if 'Unnamed: 0.1' in df.columns:
            df = df.drop(['Unnamed: 0.1'], axis=1)
        train = df[df.season != test_year]
        print(df.columns)
        X_train = train.drop(['driver', 'podium'], axis=1)
        y_train = train.podium

        self.linear_scaler = StandardScaler()
        X_train = pd.DataFrame(self.linear_scaler.fit_transform(
            X_train), columns=X_train.columns)

        if self.linear:
            res = self.train_linear_regression(
                df, X_train, y_train, self.linear_scaler, test_year=test_year)
            self.linear_model = res[0]
            self.preds = res[1]
        # if self.svm:
        #     res = self.train_svm(df, X_train, y_train, scaler, test_year=test_year)
        #     self.svm_model = res[0]
        #     self.preds = res[1].merge(
        #         self.preds, on=["round", "podium"], suffixes=('_svm', '_linear'))
        # if self.forest:
        #     res = self.train_forest(df, X_train, y_train, scaler, test_year=test_year)
        #     self.forest_model = res[0]
        #     self.preds = res[1].merge(self.preds, on=["round", "podium"])
        # if self.logit:
        #     won = [1 if x == 1 else 0 for x in y_train]
        #     res = self.train_logistic_regression(df, X_train, won, scaler, test_year = test_year)
        #     self.logit_model = res[0]
        #     self.preds = res[1].merge(self.preds, on = ["round", "podium"])
        #     # self.preds.rename("
        if self.lasso:
            res = self.train_lasso_regression(df, X_train, y_train, StandardScaler(), test_year=test_year)
            self.lasso_model = res[0]
            self.preds = res[1]
        if self.lassocv:
            lasso_scaler = StandardScaler()
            res = self.train_lassocv_regression(df, X_train, y_train, lasso_scaler, test_year=test_year)
            self.lassocv_model = res[0]
            self.preds = res[1]
        if self.lassolarscv:
            res = self.train_lassolarscv_regression(df, X_train, y_train, StandardScaler(), test_year=test_year)
            self.lassolarscv_model = res[0]
            self.preds = res[1]


    def save_linear(self, name):
        """
        save linear model with name name
        """
        with open(name, 'wb') as f:
            pickle.dump(self.linear_model, f)

    def load_linear(self, name):
        """
        load linear model with name name
        """
        with open(name, 'rb') as f:
            self.linear_model = pickle.load(f)
            self.linear_score = 0

    def save_svm(self, name):
        """
        save svm model with name name
        """
        with open(name, 'wb') as f:
            pickle.dump(self.svm_model, f)

    def load_svm(self, name):
        """
        load svm model with name name
        """
        with open(name, 'rb') as f:
            self.svm_model = pickle.load(f)
            self.svm_score = 0

    def save_forest(self, name):
        """
        save forest model with name name
        """
        with open(name, 'wb') as f:
            pickle.dump(self.forest_model, f)

    def load_forest(self, name):
        """
        load forest model with name name
        """
        with open(name, 'rb') as f:
            self.forest_model= pickle.load(f)
            self.forest_score = 0

    def save_logit(self, name):
        """
        save logistic model with name name
        """
        with open(name, 'wb') as f:
            pickle.dump(self.logit_model, f)

    def save_logit(self, name):
        """
        load logistic model with name name
        """
        with open(name, 'rb') as f:
            self.logit_model = pickle.load(f)
            self.logit_score = 0
    
    def make_zeroed_lassocv_coefs(self):
        for i in range(len(self.get_lassocv_coefs())):
            if self.get_lassocv_coefs()[i] == 0.0:
                self.zeroed_coefs += "Feature: " + self.get_lassocv_features()[i]
                self.zeroed_coefs += " Coeff: " + self.get_lassocv_coefs()[i]
    

    def make_non_zeroed_lassocv_coefs(self):
        for i in range(len(self.get_lassocv_coefs())):
            if self.get_lassocv_coefs()[i] != 0.0:
                self.non_zeroed_coefs += "Feature: " + str(self.get_lassocv_features()[i])
                self.non_zeroed_coefs += ", Coeff: " + str(self.get_lassocv_coefs()[i])