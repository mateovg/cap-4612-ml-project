# this file prepares the data for modeling
# data is read in from the kaggle db or a csv file if it exists
# the data is cleaned and elo ratings are calculated for each team
# rolling averages are calculated for each team
# the data is saved to a csv file
# it takes the number of games to use for the rolling average as a command line argument
# the default is 5 games

import pandas as pd
import sqlite3
import elo
import sys


def import_data():
    # Import data from the large sqlite database and save it as a csv file
    # to avoid needing the large database
    connection = sqlite3.connect('data/nba.sqlite')
    df_games = pd.read_sql(
        'SELECT * FROM game WHERE season_type = "Regular Season"', connection)
    df_games.to_csv('data/games.csv', index=False)


def read_data():
    # read in the csv file
    df_games = pd.read_csv('data/games.csv')
    return df_games


def clean_data(df_games):
    # couple things to prepare the data for modeling
    df_games['game_date'] = pd.to_datetime(df_games['game_date'])

    df_games['wl_home'] = df_games['wl_home'].apply(
        lambda x: 1 if x == 'W' else 0)
    df_games.rename(columns={'wl_home': 'target'}, inplace=True)

    drop_cols = ['matchup_home', 'matchup_away', 'min',
                 'video_available_home', 'video_available_away', 'wl_away', 'season_type']
    df_games.drop(columns=drop_cols, inplace=True)
    return df_games


def calculate_elo(df_games):
    elo_dict = {}
    starting_elo = 1300

    # initialize elo ratings for each team to 1500
    # should be 1300 but 1500 worked better to match my ratings with 538's
    for team in df_games['team_name_home'].unique():
        elo_dict[team] = starting_elo

    # create a new column for the elo rating of each team
    df_games['elo_home'] = starting_elo
    df_games['elo_away'] = starting_elo
    start_season = df_games['season_id'].iloc[0]

    seasons = df_games['season_id'].unique()
    for season in seasons:
        # decay elo ratings
        if season != start_season:
            for team in elo_dict:
                elo_dict[team] = elo.find_decay_elo(elo_dict[team])
        season_games = df_games[df_games['season_id'] == season]
        elo.find_elo_ratings(season_games, elo_dict)
        df_games[df_games['season_id'] == season] = season_games
    return df_games


def calculate_rolling(df_games, num_games):
    # drop rows with missing values, older games without data for 3pt, etc
    df_games.dropna(inplace=True)

    df_games.sort_values(by='game_date', inplace=True)
    df_games.reset_index(drop=True, inplace=True)

    cols = df_games.columns
    home_cols = [col for col in cols if 'away' not in col]
    away_cols = [col for col in cols if 'home' not in col]

    df_games_home = df_games[home_cols]
    df_games_away = df_games[away_cols]

    # remove suffixes on column names
    df_games_home.columns = [col.replace('_home', '')
                             for col in df_games_home.columns]
    df_games_away.columns = [col.replace('_away', '')
                             for col in df_games_away.columns]

    def calculate_rolling(df, num_games):
        ignored_cols = ['season_id', 'team_id', 'team_abbreviation',
                        'team_name', 'game_id', 'game_date', 'target', 'elo']
        rolling_cols = [col for col in df.columns if col not in ignored_cols]

        # make sure the columns are in the correct order
        df.sort_values(by='game_date', inplace=True)

        # group by team and season, create new df to store averages
        grouped_df = df.groupby(['season_id', 'team_id'])
        rolling_df = df.copy()

        for col in rolling_cols:
            # on each column, calculate the rolling average
            # the columns are grouped by team and season
            # closed="left" means only previous rows are used to calculate average
            # reset_index makes it a regular column
            rolling_df[col] = grouped_df[col].rolling(num_games, min_periods=1, closed="left").mean(
            ).round(decimals=2).reset_index(drop=True)

        # Sort the rolling DataFrame back based on 'game_date'
        rolling_df.sort_values(by='game_date', inplace=True)

        # drop the first row since it will be NaN
        rolling_df.dropna(inplace=True)

        return rolling_df

    df_games_home_rolling = calculate_rolling(df_games_home, num_games)
    df_games_away_rolling = calculate_rolling(df_games_away, num_games)

    # now we rejoin the home and away dataframes on the game_id
    df_games_rolling = df_games_home_rolling.merge(df_games_away_rolling, on=[
        'game_id', 'season_id', 'game_date'], suffixes=('_home', '_away'))

    # rename the target column to be the target for the home team, remove the away target
    df_games_rolling.rename(columns={'target_home': 'wl_home'}, inplace=True)
    df_games_rolling.drop(columns=['target_away'], inplace=True)

    # drop na values
    df_games_rolling.dropna(inplace=True)

    return df_games_rolling


def drop_seasons(df_games):
    # drop seasons from the df if they have fewer than 100 games
    # this is to avoid the first few seasons where there were fewer teams
    seasons = df_games['season_id'].unique()
    for season in seasons:
        season_games = df_games[df_games['season_id'] == season]
        if len(season_games) < 100:
            df_games = df_games[df_games['season_id'] != season]
    return df_games


def prepare_data(num_games=5):
    # import data if csv file doesn't exist
    try:
        print('Reading data...')
        df_games = read_data()
    except:
        print('Importing data...')
        import_data()
        df_games = read_data()

    print('Cleaning data...')
    df_games = clean_data(df_games)

    # calculate elo ratings for each team
    print('Calculating elo ratings...')
    df_games = calculate_elo(df_games)

    # calculate rolling averages for each team
    print(f'Calculating rolling averages with {num_games} game window...')
    df_games_rolling = calculate_rolling(df_games, num_games)

    # drop seasons with fewer than 100 games
    df_games_rolling = drop_seasons(df_games_rolling)

    # save the data to a csv file
    print('Saving data...')
    df_games_rolling.to_csv('data/games_rolling.csv', index=False)

    rows, cols = df_games.shape
    print(f'Finished! {rows} rows and {cols} columns')
