from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def fit_models(sample, ticker, cv=0):
    ITER_SIZE = 5
    _df_list = []

    for col in sample.columns:
        _df = pd.DataFrame()
        _df[col] = sample[col]
        for i in range(ITER_SIZE):
            _df[f"{col}_{i}"] = sample[col].shift(periods=i + 1)
        _df_list.append(_df)

    _df = pd.concat(_df_list, axis=1)

    sample = _df

    X = sample.dropna().drop(["target"], axis=1)
    X = X.dropna().drop([f"target_{i}" for i in range(ITER_SIZE)], axis=1)

    sample.dropna(inplace=True)
    y = sample[
        "target"].shift(-1).apply(lambda x: 1 if x > 0 else 0)

    scaler = StandardScaler()
    pipeline = Pipeline(steps=[("scaler", scaler), ])
    X = pipeline.fit_transform(X)

    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]

    y_train = np.stack(y_train.values.tolist(), axis=0)
    y_test = np.stack(y_test.values.tolist(), axis=0)
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    X_train_lstm = X_train.reshape(X_train.shape[0], -1, ITER_SIZE + 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], -1, ITER_SIZE + 1)

    y_preds = []

    print("\tFitting Models...")

    svm = SVC(C=1, gamma=0.1)
    svm.fit(X_train, y_train)
    y_preds.append(svm.predict(X_test).astype(int))
    # 100 for 1y data
    knn = KNeighborsClassifier(n_neighbors=100, weights="distance", algorithm="auto", leaf_size=1)
    knn.fit(X_train, y_train)
    y_preds.append(knn.predict(X_test).astype(int))

    rf = RandomForestClassifier(n_estimators=9, criterion="gini", min_samples_leaf=5, max_depth=1)
    rf.fit(X_train, y_train)
    y_preds.append(rf.predict(X_test).astype(int))

    gb = GradientBoostingClassifier(n_estimators=1, max_features=7, max_depth=1)
    gb.fit(X_train, y_train)
    y_preds.append(gb.predict(X_test).astype(int))

    xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, min_child_weight=10, gamma=0, learning_rate=0.1, seed=27, subsample=0.65)
    xgb_model.fit(X_train, y_train)
    y_preds.append(xgb_model.predict(X_test).astype(int))

    lstm = Sequential([
        layers.Input((X_train_lstm.shape[1], ITER_SIZE + 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    lstm.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    lstm.fit(X_train_lstm, y_train, epochs=100, verbose=0)


    y_preds.append((lstm.predict(X_test_lstm).flatten() >= 0.5).astype(int))

    y_test = y_test.astype(int)

    average = "binary"
    df = pd.DataFrame({"ticker": ticker,
                       "svm_acc": metrics.accuracy_score(y_test, y_preds[0]),
                       "svm_prec": metrics.precision_score(y_test, y_preds[0], average=average),
                       "svm_recall": metrics.recall_score(y_test, y_preds[0], average=average),
                       "svm_f1": metrics.f1_score(y_test, y_preds[0], average=average),
                       "knn_acc": metrics.accuracy_score(y_test, y_preds[1]),
                       "knn_prec": metrics.precision_score(y_test, y_preds[1], average=average),
                       "knn_recall": metrics.recall_score(y_test, y_preds[1], average=average),
                       "knn_f1": metrics.f1_score(y_test, y_preds[1], average=average),
                       "rf_acc": metrics.accuracy_score(y_test, y_preds[2]),
                       "rf_prec": metrics.precision_score(y_test, y_preds[2], average=average),
                       "rf_recall": metrics.recall_score(y_test, y_preds[2], average=average),
                       "rf_f1": metrics.f1_score(y_test, y_preds[2], average=average),
                       "gb_acc": metrics.accuracy_score(y_test, y_preds[3]),
                       "gb_prec": metrics.precision_score(y_test, y_preds[3], average=average),
                       "gb_recall": metrics.recall_score(y_test, y_preds[3], average=average),
                       "gb_f1": metrics.f1_score(y_test, y_preds[3], average=average),
                       "xgb_acc": metrics.accuracy_score(y_test, y_preds[4]),
                       "xgb_prec": metrics.precision_score(y_test, y_preds[4], average=average),
                       "xgb_recall": metrics.recall_score(y_test, y_preds[4], average=average),
                       "xgb_f1": metrics.f1_score(y_test, y_preds[4], average=average),
                       "lstm_acc": metrics.accuracy_score(y_test, y_preds[5]),
                       "lstm_prec": metrics.precision_score(y_test, y_preds[5], average=average),
                       "lstm_recall": metrics.recall_score(y_test, y_preds[5], average=average),
                       "lstm_f1": metrics.f1_score(y_test, y_preds[5], average=average), }, index=[0])


    return df


def plot_scores(score_df):
    columns_to_plot = ["svm_acc", "knn_acc", "rf_acc", "gb_acc", "xgb_acc", "lstm_acc"]

    ticker = score_df["ticker"].values[0]
    scores = score_df[columns_to_plot].values[0]

    plt.figure(figsize=(10, 8))
    plt.bar(columns_to_plot, scores, color='blue', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Scores for Ticker: {ticker}')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
