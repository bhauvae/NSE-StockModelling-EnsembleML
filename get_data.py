import yfinance as yf
import pickle
import time
import requests
import bs4 as bs
import pandas as pd


def get_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_National_Stock_Exchange_of_India")
    soup = bs.BeautifulSoup(resp.text, "html.parser")
    stocks = []
    table = soup.find_all(class_="external text")

    for tag in table[6:-2]:
        stocks.append(tag.text + ".NS")

    pd.to_pickle(pd.Series(stocks), "./stocks.pkl")

    return stocks


def save_data():
    tickers = pd.read_pickle("./NSE/stocks.pkl")
    tickers = tickers.values

    counter = 0

    for i in range(0, len(tickers), 12):
        ticker_list = tickers[i : i + 12]
        ticker_list = ",".join(ticker_list)
        data = yf.download(
            tickers=ticker_list,
            threads=True,
            group_by='ticker',
            period="1y"

        )

        counter = counter + 1
        data = data.T
        pd.to_pickle(data, f"./NSE/1y/data/{counter}.pkl")


save_data()
