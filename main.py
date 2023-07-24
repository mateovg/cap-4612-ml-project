# Main file for running the NBA predictor and generating data for final report

# Data from Kaggle: https://www.kaggle.com/datasets/wyattowalsh/basketball


import time
import pandas as pd
from nba_predictor import NBAPredictor

observations = []


def test_models(num_games, num_features, no_elo=False):
    predictor = NBAPredictor(
        num_games=num_games, num_features=num_features, no_elo=no_elo)
    lr_dict = predictor.log_reg_model().copy()
    knn_dict = predictor.knn_model().copy()

    lr_dict['model'] = 'Logistic Regression'
    knn_dict['model'] = 'KNN'

    lr_dict['num_games'] = num_games
    knn_dict['num_games'] = num_games

    lr_dict['num_features'] = num_features
    knn_dict['num_features'] = num_features

    lr_dict['no_elo'] = no_elo
    knn_dict['no_elo'] = no_elo

    observations.append(lr_dict)
    observations.append(knn_dict)


def final_report():
    start_time = time.time()

    predictor = NBAPredictor(num_games=5, num_features=5)
    observations.append(predictor.base_line().copy())
    observations.append(predictor.elo_model().copy())
    observations.append(predictor.elo_model(homecourt_adv=0).copy())

    test_models(5, 5)
    test_models(10, 5)
    test_models(5, 10)
    test_models(15, 10)
    test_models(5, 5, no_elo=True)

    df = pd.DataFrame(observations)
    df.to_csv('observations.csv', index=False)

    print(f'Finished in {time.time() - start_time} seconds')


if __name__ == "__main__":
    # final_report()
    predictor = NBAPredictor(num_games=5, num_features=5)
    predictor.elo_model(homecourt_adv=0)
    predictor.elo_model()
    predictor.log_reg_model()
