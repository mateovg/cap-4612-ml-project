import pandas as pd
# import sqlite3


def read_in():
    # connection = sqlite3.connect('data/nba.sqlite')
    # cursor = connection.cursor()

    # cursor.execute('DROP TABLE IF EXISTS play_by_play')
    # games = pd.read_sql('SELECT * FROM game WHERE season_type = "Regular Season"', connection)
    # export = games.to_csv('data/games.csv', index=False)

    games = pd.read_csv('data/games.csv')

    games.head()

    # convert date to datetime
    games['game_date'] = pd.to_datetime(games['game_date'])

    # change wl_home to 1 if home team won, 0 if home team lost
    games['wl_home'] = games['wl_home'].apply(lambda x: 1 if x == 'W' else 0)
    games.rename(columns={'wl_home': 'target'}, inplace=True)

    # drop columns that won't be used
    drop_cols = ['matchup_home', 'matchup_away', 'min',
                 'video_available_home', 'video_available_away', 'wl_away', 'season_type']
    games = games.drop(columns=drop_cols).copy()
    return games

# functions to calculate each part of the elo formula


def find_K(MOV, elo_diff):
    # positive MOV means home team won
    K_0 = 20
    if MOV > 0:
        multiplier = (MOV + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)
    else:
        multiplier = (-MOV + 3) ** 0.8 / (7.5 + 0.006 * -elo_diff)
    # returns two values, one for each team
    return K_0 * multiplier, K_0 * multiplier


def find_S(MOV):
    if MOV > 0:  # home team won
        return 1, 0
    elif MOV < 0:  # home team lost
        return 0, 1
    else:
        return 0.5, 0.5  # tie


def find_E(elo_home, elo_away):
    home_adv = 100
    elo_home += home_adv

    E_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400.0))
    E_away = 1 / (1 + 10 ** ((elo_home - elo_away) / 400.0))

    return E_home, E_away


def find_new_elo(elo_home, elo_away, MOV):
    elo_diff = elo_home - elo_away
    K_home, K_away = find_K(MOV, elo_diff)
    S_home, S_away = find_S(MOV)
    E_home, E_away = find_E(elo_home, elo_away)

    elo_home = elo_home + K_home * (S_home - E_home)
    elo_away = elo_away + K_away * (S_away - E_away)

    return elo_home, elo_away


def find_decay_elo(elo):
    return elo * 0.75 + 1505 * 0.25


def create_elo_columns(games):
    # create a dictionary to store the elo ratings for each team
    elo_dict = {}
    starting_elo = 1300

    # initialize elo ratings for each team to 1500
    # should be 1300 but 1500 worked better to match my ratings with 538's
    for team in games['team_name_home'].unique():
        elo_dict[team] = starting_elo

    # create a new column for the elo rating of each team
    games['elo_home'] = starting_elo
    games['elo_away'] = starting_elo
    start_season = games['season_id'].iloc[0]
    current_season = start_season

    # loop through each game and calculate the new elo rating for each team
    # each row should have the elo rating BEFORE the game

    def find_elo_ratings(season):
        for index, row in season.iterrows():
            home_team = row['team_name_home']
            away_team = row['team_name_away']

            elo_home = elo_dict[home_team]
            elo_away = elo_dict[away_team]

            season.loc[index, 'elo_home'] = elo_home
            season.loc[index, 'elo_away'] = elo_away

            MOV = row['plus_minus_home']

            elo_home_new, elo_away_new = find_new_elo(elo_home, elo_away, MOV)

            elo_dict[home_team] = elo_home_new
            elo_dict[away_team] = elo_away_new

    # loop through each season and calculate the elo ratings for each game
    # if it's not the first season, decay the elo ratings
    seasons = games['season_id'].unique()
    for season in seasons:
        # decay elo ratings
        if season != start_season:
            for team in elo_dict:
                elo_dict[team] = find_decay_elo(elo_dict[team])
        season_games = games[games['season_id'] == season]
        find_elo_ratings(season_games)
        games[games['season_id'] == season] = season_games


games = read_in()
create_elo_columns(games)

# drop rows with missing values, older games without data for 3pt, etc
games.dropna(inplace=True)

games.sort_values(by='game_date', inplace=True)
games.reset_index(drop=True, inplace=True)

cols = games.columns
home_cols = [col for col in cols if 'away' not in col]
away_cols = [col for col in cols if 'home' not in col]

games_home = games[home_cols]
games_away = games[away_cols]

games_home.columns = [col.replace('_home', '') for col in games_home.columns]
games_away.columns = [col.replace('_away', '') for col in games_away.columns]


def calculate_rolling(df, num_games):
    ignored_cols = ['season_id', 'team_id', 'team_abbreviation', 'team_name', 'game_id',
                    'game_date', 'target', 'elo']

    # make sure the columns are in the correct order
    df.sort_values(by='game_date', inplace=True)

    for col in df.columns:
        if col not in ignored_cols:
            # replace the column with the rolling average
            df.loc[:, col] = df[col].rolling(
                num_games, min_periods=2).mean().round(decimals=2)
            # # rename the column to indicate it's a rolling average
            # df.rename(columns={col: col + '_rolling_' + str(num_games)}, inplace=True)

    return df


games_home_rolling = calculate_rolling(games_home, 10)
games_away_rolling = calculate_rolling(games_away, 10)

# now we rejoin the home and away dataframes on the game_id
games_rolling = games_home_rolling.merge(games_away_rolling, on=[
                                         'game_id', 'season_id', 'game_date'], suffixes=('_home', '_away'))

# rename the target column to be the target for the home team
games_rolling.rename(columns={'target_home': 'wl_home'}, inplace=True)
games_rolling.rename(columns={'target_away': 'wl_away'}, inplace=True)

# drop na values
games_rolling.dropna(inplace=True)

# export data to a csv file
games_rolling.to_csv('data/games_rolling.csv', index=False)
