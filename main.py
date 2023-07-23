from prepare_data import prepare_data
from nba_predictor import knn_model, log_reg_model

# prepare data with 5 game rolling average
prepare_data(num_games=5)

# run the model
# print('Running knn model...')
# knn_model(k=15)

print('Running logistic regression model...')
log_reg_model(10)
