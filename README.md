# Predicting the winner of NBA games using machine learning

This project aims to predict the outcome of NBA games using machine learning models. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball).

The data is preprocessed with the `prepare_data.py` program, with functions form `elo.py`, and used by the `NBAPredictor` class in `nba_predictor.py` in order to train and run models on the data. The models iteratively trained on previous seasons, starting at the 3rd season of the dataset, until the current one and all their predicted values are used to measure performance.

## How to use

The `main.py` file has all you need to run the model. Simply create a `NBAPredictor` object with the number of games to include in the rolling average and number of features to select in the models as arguments `num_games` and `num_features` respectively. 

To run a model, call the appropriate method on the `NBAPredictor` object. `elo_model(homework_adv=100)` and `base_line()` will run simple baseline models using only the greater elo rating or the home team as the predictors. `log_reg_model` will train and run a logistic regression model and `knn_model(k=neighbors)` will train and run a k-nearest neighbor model.

All these methods will print their metrics as well as return a python dictionary with the relevant data.

## Needed libraries

`pandas`, `scikit-learn`, `sqlite3`