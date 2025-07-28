## Shell
The problem statement was kaggle style and asked to build a regression model for prediction blending fuel prices. The output was supposed to be in csv format. We build the model with 90% correctness score and were ranked 83 out of 6K participating teams across the nation.
Our approach was building a basic XGboost model and ensembling it with other models LightBGM . Our Final MAPE score corresponing to the leaderboard score was 0.35, 0.01% on the leaderboard
