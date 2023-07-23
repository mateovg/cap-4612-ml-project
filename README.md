# Predicting the winner of NBA games using machine learning

## How to use

The `main.py` file has all you need to run the model. 

You can choose the number of games to use when calculating a team's average stats with the `prepare_data()` function's `num_games` argument.

Then you can choose which model to run the K-nearest neighbor or logistic regression model with the `knn_model` and `log_reg_model` functions from the `nba_predictor.py` file. `knn_model` takes an argument `k` for the amount of neighbors to use and both functions take an unspecified argument for the number of features to select for with a sequential feature selector.



The four major performance metrics will be printed out for each model.

## 
```
python main.py out.txt
```