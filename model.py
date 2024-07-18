import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import numpy as np
from datetime import date, timedelta
import plotly.graph_objs as go

def prediction(stock, n_days):
    # Load the data
    df = yf.download(stock, period='1mo')
    df.reset_index(inplace=True)
    df['Day'] = df.index

    # Splitting the dataset
    X = df[['Day']]
    Y = df[['Close']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # GridSearch for SVR
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"])

    rbf_svr = best_svr
    rbf_svr.fit(x_train, y_train)

    # Forecasting future days
    last_day = df['Day'].iloc[-1]
    output_days = np.arange(last_day + 1, last_day + n_days).reshape(-1, 1)

    # Generating future dates
    last_date = df['Date'].iloc[-1]
    dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=rbf_svr.predict(output_days), mode='lines+markers', name='Predicted Close'))
    fig.update_layout(
        title="Predicted Close Price for the next " + str(n_days) + " days",
        xaxis_title="Date",
        yaxis_title="Close Price",
    )

    return fig
