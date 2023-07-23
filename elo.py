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
    # 1505 to aim for a 1500 mean
    # due to new teams joining the league
    return elo * 0.75 + 1505 * 0.25


def find_elo_ratings(season, elo_dict):
    # loop through each game and calculate the new elo rating for each team
    # each row should have the elo rating BEFORE the game

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
