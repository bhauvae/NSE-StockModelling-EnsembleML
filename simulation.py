import pandas as pd

from format_data import format_data
from create_models import fit_models


def simulation(dataset, i):
    score = pd.DataFrame()

    tickers = pd.Series(dataset.index.get_level_values(0).values).drop_duplicates()

    for ticker in tickers.values:
        print(ticker)
        data = dataset.loc[ticker].T
        data.index = pd.to_datetime(data.index).date
        data.dropna(inplace=True)

        # if data is empty pass
        if not data.empty:
            data = format_data(data)
            try:
                data_score = fit_models(data, ticker)
                score = pd.concat([score, data_score])
            except:
                continue

    pd.to_pickle(score, f"./NSE/1y/score/{i}.pkl")


def run():
    for i in range(1, 152):
        dataset = pd.read_pickle(f"./NSE/1y/data/{i}.pkl")
        simulation(dataset, i)
        print(f"{i} COMPLETE")


run()
