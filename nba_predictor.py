# data normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# time series split for cross validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector

# classifiers we will use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd


def get_data():
    # simply reads the prepared data from a csv file
    df_games = pd.read_csv('data/games_rolling.csv')
    df_games = df_games.select_dtypes(include=['float64', 'int64'])
    return df_games


def find_metrics(true, pred):
    # find the metrics for the binary classification model

    # some bitwise operations to find the metrics
    true_positives = ((true == 1) & (pred == 1)).sum()
    false_positives = ((true == 0) & (pred == 1)).sum()
    true_negatives = ((true == 0) & (pred == 0)).sum()
    false_negatives = ((true == 1) & (pred == 0)).sum()
    total = true_positives + false_positives + true_negatives + false_negatives

    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def print_metrics(labels, pred):
    metrics = find_metrics(labels, pred)
    print(f"Accuracy: {metrics['accuracy']:0.4f}")
    print(f"Precision: {metrics['precision']:0.4f}")
    print(f"Recall: {metrics['recall']:0.4f}")
    print(f"F1 Score: {metrics['f1']:0.4f}")


def back_test(data, model, features):
    # test the model on previous seasons iteratively
    # each iteration, the model is trained on all previous seasons
    # and tested on the current season
    target = 'wl_home'
    seasons = data['season_id'].unique()

    all_predictions = pd.DataFrame(columns=['actual', 'predicted'])

    for i in range(2, len(seasons)):
        # start on the 3rd season
        season = seasons[i]
        train = data[data['season_id'] < season]
        test = data[data['season_id'] == season]

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        season_predictions = pd.DataFrame(
            {'actual': y_test, 'predicted': predictions}
        )
        all_predictions = pd.concat([all_predictions, season_predictions])

    return all_predictions


def find_best_features(data, model, n_features):
    # find the best features using sequential feature selector
    ignored_cols = ['season_id', 'team_id_home',
                    'game_id', 'team_id_away', 'wl_home']
    feature_cols = data.columns[~data.columns.isin(ignored_cols)]
    target_col = 'wl_home'

    # 5 fold cross validation for time series data
    split = TimeSeriesSplit(n_splits=5)
    # finds the best features using sequential feature selector
    sfs = SequentialFeatureSelector(
        model, n_features_to_select=n_features, cv=split, n_jobs=-1)

    # Create a copy of the data to avoid modifying the original data
    data_copy = data.copy()

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    data_copy[feature_cols] = scaler.fit_transform(data_copy[feature_cols])

    # fit the model
    sfs.fit(data_copy[feature_cols], data_copy[target_col])

    return list(feature_cols[sfs.get_support()])


def evaluate(data, model, n_features):
    print(f'Finding best {n_features} features...')
    best_features = find_best_features(data, model, n_features)
    print(f'Best features: {best_features}')

    print('Back testing model...')
    predictions = back_test(data, model, best_features)

    print('Metrics:')
    actual, predicted = predictions['actual'], predictions['predicted']
    print_metrics(actual, predicted)


def knn_model(features, k=10):
    data = get_data()
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    evaluate(data, model, features)


def log_reg_model(features):
    data = get_data()
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    evaluate(data, model, features)


def elo_model():
    data = get_data()
    true = data['wl_home']
    pred = data['elo_home'] + 100 > data['elo_away']
    print_metrics(true, pred)


def base_line():
    data = get_data()
    true = data['wl_home']
    pred = 1
    print_metrics(true, pred)
