# Walmart-Closing-Stock-Price-Forecasting

# Forecasting 
The NYSE prices dataset is a time series data set of 851264 records depicting the daily prices of stocks for 501 companies, spanning from years 2010 to 2016.
We choose to forecast closing stock price of Walmart. It had 1762 records with 7 columns.

We couldn’t do the regular predictions on this kind of data, as the stock market rapidly changes every day, it makes no sense if we are predicting todays stock value based on stocks of last 5 years .
So we are predicting a particular day’s closing stock value based on past five days. 

We used Linear Regression to train the model and it was trained with an accuracy of 83.97 % .

# predicting earning per share and classifying bankruptcy
The NYSE fundamentals data set has 1781 records with 77  feature metrics from annual SEC 10K fillings (2012-2016) .
We started of with Min-Max scaling and since there were huge number of features, we had to scale them down. 

We used Radom Forests classifier for feature selection, since random forests considers correlated features as well, to over come this, we used co-relational matrix to trim out highly correlated features. We successfully got the 77 features down to 31  features. 

We used Random Forest Regressor for predicting earnings per share and training model was 80.94 %.

Using K Means Clustering, we classified 2 companies F and VZ that might go bankrupt in near future, using features like Total Assets, Gross Profit, Net Income, Total Equity, Long Term Debt.

![Poster](/Users/apple/Desktop/Screen\ Shot\ 2020-08-23\ at\ 10.28.24\ AM.png)
