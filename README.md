# Walmart-Closing-Stock-Price-Forecasting

The NYSE prices dataset is a time series data set of 851264 records depicting the daily prices of stocks for 501 companies, spanning from years 2010 to 2016.
We choose to forecast closing stock price of Walmart. It had 1762 records with 7 columns.

We couldn’t do the regular predictions on this kind of data, as the stock market rapidly changes every day, it makes no sense if we are predicting todays stock value based on stocks of last 5 years .
So we are predicting a particular day’s closing stock value based on past five days. 

We used Linear Regression to train the model and it was trained with an accuracy of 83.97 % .
