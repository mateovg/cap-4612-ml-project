# data normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# time series split for cross validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector

# classifiers we will use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import prepare_data

target_col = 'wl_home'
ignored_cols = ['season_id', 'team_id_home',
                'game_id', 'team_id_away', 'wl_home']


class NBAPredictor:

    def __init__(self, num_games, num_features, no_elo=False):
        self.num_games = num_games
        self.num_features = num_features
        self.data = self.get_data()

        self.ignored_cols = ignored_cols

        if no_elo:
            self.ignored_cols += ['elo_home', 'elo_away']

        self.feature_cols = self.data.columns[~self.data.columns.isin(
            self.ignored_cols)]

        self.best_features = None
        self.true = self.data[target_col]
        self.pred = None
        self.confusion_matrix = None

        self.model = None

        self.info = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'model': self.model,
            'features': self.best_features,
            'num_games': num_games,
            'cm': self.confusion_matrix
        }

    def get_data(self):
        # try to open a csv file with the rolling averages if it exists
        # otherwise, prepare the data
        try:
            df_games = pd.read_csv(
                f'data/games_rolling_{self.num_games}.csv')
            df_games = df_games.select_dtypes(include=['float64', 'int64'])
            return df_games
        except:
            return self.update_data()

    def update_data(self):
        prepare_data.prepare_data(num_games=self.num_games)
        return self.get_data()

    def update_info(self):
        self.info['model'] = self.model
        self.info['features'] = self.best_features
        self.info['num_games'] = self.num_games
        self.calculate_metrics()

    def display_info(self):
        print("Model Info:")
        print("-----------")
        print(f"Model: {self.info['model']}")
        print(f"Number of Games: {self.info['num_games']}")
        print(f"Best Features: {', '.join(self.info['features'])}")
        print("Metrics:")
        print(f"Accuracy: {self.info['accuracy']:0.4f}")
        print(f"Precision: {self.info['precision']:0.4f}")
        print(f"Recall: {self.info['recall']:0.4f}")
        print(f"F1 Score: {self.info['f1']:0.4f}")

    def calculate_metrics(self):
        # find the metrics for the binary classification model

        # some bitwise operations to find the metrics
        true_positives = ((self.true == 1) & (self.pred == 1)).sum()
        false_positives = ((self.true == 0) & (self.pred == 1)).sum()
        true_negatives = ((self.true == 0) & (self.pred == 0)).sum()
        false_negatives = ((self.true == 1) & (self.pred == 0)).sum()
        total = true_positives + false_positives + true_negatives + false_negatives

        self.confusion_matrix = {
            'tp': true_positives,
            'fp': false_positives,
            'tn': true_negatives,
            'fn': false_negatives
        }

        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

        self.info['accuracy'] = accuracy
        self.info['precision'] = precision
        self.info['recall'] = recall
        self.info['f1'] = f1

    def get_metrics(self):
        self.calculate_metrics()
        return self.info

    def back_test(self):
        # test the model on previous seasons iteratively
        # each iteration, the model is trained on all previous seasons
        # and tested on the current season
        target = 'wl_home'
        seasons = self.data['season_id'].unique()

        all_predictions = pd.DataFrame(columns=['actual', 'predicted'])

        for i in range(2, len(seasons)):
            # start on the 3rd season
            season = seasons[i]
            train = self.data[self.data['season_id'] < season]
            test = self.data[self.data['season_id'] == season]

            X_train, y_train = train[self.best_features], train[target]
            X_test, y_test = test[self.best_features], test[target]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            season_predictions = pd.DataFrame(
                {'actual': y_test, 'predicted': predictions}
            )
            all_predictions = pd.concat([all_predictions, season_predictions])

        self.pred = all_predictions['predicted']
        self.true = all_predictions['actual']

    def find_best_features(self):
        split = TimeSeriesSplit(n_splits=10)
        # finds the best features using sequential feature selector
        sfs = SequentialFeatureSelector(
            self.model, n_features_to_select=self.num_features, cv=split, n_jobs=-1)

        # Create a copy of the data to avoid modifying the original data
        data_copy = self.data.copy()

        # Normalize the data using standard scaler
        scaler = StandardScaler()
        data_copy[self.feature_cols] = scaler.fit_transform(
            data_copy[self.feature_cols])

        # fit the model
        sfs.fit(data_copy[self.feature_cols], data_copy[target_col])

        self.best_features = list(self.feature_cols[sfs.get_support()])

    def evaluate(self):
        self.pred = None
        print(f'Finding best {self.num_features} features...')
        self.find_best_features()
        print(f"Best Features: {', '.join(self.best_features)}")

        print('Back testing model...')
        self.back_test()

        self.update_info()
        print('Done!')
        self.display_info()

    def knn_model(self, k=10):
        print(
            f'Running knn model with {k} neighbors, {self.num_games} game rolling average, and {self.num_features} features...')
        self.model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        self.evaluate()
        return self.info

    def log_reg_model(self):
        print(
            f'Running logistic regression model, {self.num_games} game rolling average, and {self.num_features} features...')
        self.model = LogisticRegression(max_iter=1000, n_jobs=-1)
        self.evaluate()
        return self.info

    def base_line(self):
        self.true = self.data['wl_home']
        self.pred = self.true.any()
        self.model = 'baseline'
        self.best_features = []
        self.update_info()
        self.display_info()
        return self.info

    def elo_model(self, homecourt_adv=100):
        self.true = self.data['wl_home']
        self.pred = self.data['elo_home'] + \
            homecourt_adv > self.data['elo_away']
        self.model = 'elo'
        self.best_features = []
        self.update_info()
        self.display_info()
        return self.info
