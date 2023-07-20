import sqlite3
import pandas as pd

# imports for models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# connect to db
connection = sqlite3.connect('data/nba.sqlite')
# df of all games
df_games = pd.read_sql(
    'select * from game where season_id > 22010 and season_type = "Regular Season"', connection)

# remove some columns we don't care about
remove_columns = ['team_abbreviation_home', 'team_name_home', 'game_date', 'matchup_home', 'min',
                  'video_available_home', 'team_abbreviation_away', 'team_name_away', 'matchup_away',
                  'wl_away', 'video_available_away', 'season_type']
df_games.drop(remove_columns, axis=1, inplace=True)

# make wl_home a binary column
df_games['wl_home'] = df_games['wl_home'].apply(lambda x: 1 if x == 'W' else 0)

# split up the home team and away team columns
columns = df_games.columns.tolist()

# home columns won't have "away" in them and vice-versa
home_columns = [column for column in columns if 'away' not in column]
away_columns = [column for column in columns if 'home' not in column]

# create df for home and away teams
df_home = df_games[home_columns]
df_away = df_games[away_columns]

# Calculate rolling average for home and away teams
rolling_window = 10


def calculate_rolling(df, type):
    ignored_cols = [f'team_id_{type}', 'season_id', 'game_id']
    if type == "home":
        ignored_cols.append('wl_home')
    # make sure games are in temporal order
    df.sort_values([f'team_id_{type}', 'season_id', 'game_id'], inplace=True)

    # we don't want the average to carry across seasons
    rolling_average = df.groupby([f'team_id_{type}', 'season_id']).rolling(
        window=rolling_window, min_periods=1).mean().reset_index()

    # return pd.merge(df, rolling_average, on=[f'team_id_{type}', 'season_id', 'game_id'])
    return pd.concat([df[ignored_cols], rolling_average.drop(ignored_cols, axis=1)], axis=1)


df_home_rolling = calculate_rolling(df_home, "home")
df_away_rolling = calculate_rolling(df_away, "away")

# Join the home and away rolling average dataframes on game_id
df_rolling_average = df_home_rolling.merge(
    df_away_rolling, on='game_id', suffixes=('_home', '_away'))

# split the data into features and labels
# drop any columns that is object type
X = df_rolling_average.drop(
    ['game_id', 'wl_home', 'team_id_home', 'team_id_away'], axis=1)
y = df_rolling_average['wl_home']

# function for showing measurements of a model


def model_results(model, actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    print(f"Scores for {model} model: ")
    print(
        f"Accuracy: {accuracy:.2f} \t Precision: {precision:.2f}\nRecall: {recall:.2f}\t\t F1: {f1:.2f}")


# baseline results, just guessing the home team wins every time
model_results("baseline", df_games['wl_home'], [1] * len(df_games))
