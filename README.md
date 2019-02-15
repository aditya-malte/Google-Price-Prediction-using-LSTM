Google Price prediction using historical data 
Window-60 days
Used 3 features to predict price(Closing price, etc).
I have used historical data and trained an LSTM neural network on the dataset.
Approach:
1) Preprocessing the dataset
2) Splitting the dataset per time-stamp(per day) where the output is the stock price of the 
next day.
Future additions:
1) Addding financial metrics like RSI, MACD, etc. to detect trends 
   in graph.
2) Try training it in reverse too(i.e BiLSTM), to check for probable 
   performance improvements.
3) Try other time series models (autoregressive, ARIMA, etc) for this problem.
