# %%
import pandas as pd


path = "./NSE/1y/score/"
scores = pd.DataFrame()
for i in range(1,152):
    d = pd.read_pickle(path + f"{i}.pkl")
    scores = pd.concat([scores,d])
pd.to_pickle(scores,"./1y.pkl")
# # %%
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_scores(score_df):
#     columns_to_plot = ["svm_acc", "knn_acc", "rf_acc", "gb_acc", "xgb_acc", "lstm_acc"]
#
#     # ticker = score_df["ticker"].values[0]
#     scores = score_df[columns_to_plot].values[0]
#
#     plt.figure(figsize=(10, 8))
#     plt.bar(columns_to_plot, scores, color='blue', alpha=0.7)
#     plt.xlabel('Models')
#     plt.ylabel('Accuracy')
#     # plt.title(f'Accuracy Scores for Ticker: {ticker}')
#     plt.yticks(np.arange(0, 1, 0.1))
#     plt.ylim(0, 1.0)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#
#     plt.tight_layout()
#     plt.show()
# # %%
