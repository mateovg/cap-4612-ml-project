# data normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# time series split for cross validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector

# classifiers we will use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_data():
    df_games = pd.read_csv('data/games_rolling.csv')
    df_games = df_games.select_dtypes(include=['float64', 'int64'])
    return df_games


def find_metrics(labels, pred):
    return {
        'accuracy': accuracy_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred),
        'f1': f1_score(labels, pred)
    }


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

    sfs.fit(data_copy[feature_cols], data_copy[target_col])

    return list(feature_cols[sfs.get_support()])


def evaluate(data, model, n_features):
    best_features = find_best_features(data, model, n_features)
    predictions = back_test(data, model, best_features)
    actual, predicted = predictions['actual'], predictions['predicted']
    print_metrics(actual, predicted)


# pipeline(get_data(), KNeighborsClassifier(n_jobs=-1), 10)
evaluate(get_data(), LogisticRegression(n_jobs=-1, max_iter=2000), 10)
