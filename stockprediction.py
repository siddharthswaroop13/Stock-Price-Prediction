



# Introduction
# This is an attempt to predict Stock prices based on Stock prices of previous days. The stock market refers to the
# collection of markets and exchanges where regular activities of buying, selling, and issuance of shares of publicly-held companies take place.
#
# This is a time series analysis and we will see simple eight ways to predict the Stock prices. The various models to be used are:
#
# Average
# Weighted Average
# Moving Average
# Moving Weighted Average
# Linear Regression
# Weighted Linear Regression
# Lasso Regression
# Moving Window Neural Network

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

from sklearn.metrics import mean_squared_error as mse

# The Data
# The data we use for prediction would be for closing price of Infosys in NSE for the business days in 2015.
# So we will import only the Date column and Closing price column.

df = pd.read_csv("C:\\Users\\Siddharth\\Downloads\\infy_stock.csv",usecols=['Date', 'Close'], parse_dates=['Date'],index_col='Date')
print(df.head())


ts = df['Close']
print(ts.head())
print("\n")

rcParams['figure.figsize'] = 15,6
plt.plot(ts)
plt.show()

# We have data on working days only and so there are 248 data with start date as 01-01-2015 and end date as 31-12-2015.

print(df.info())
print("\n")

print("Min:",df.index.min())
print("Max:",df.index.max())
print("\n")


plt.figure(figsize=(17,5))
df.Close.plot()
plt.grid()
plt.title("Closing Price",fontsize=20)
plt.show()

# The Split
plt.figure(figsize=(17,5))
stock_price = pd.concat([df.Close[:'2015-06-12']/2,df.Close['2015-06-15':]]) # adjustment
plt.plot(stock_price)
plt.grid()
plt.title("Closing Price Adjusted",fontsize=20)
plt.show()

#print(df.Close[:'2015-06-12']/2)

print(stock_price)
print("\n")

print(stock_price[0:80])

# And now we have an adjusted time series of Infosys stock prices.
#
# Lets now Predict the Stock price based on various methods.
#
# We will predict the values on last 68 days in the series.
# We will use Mean squared error as a metrics to calculate the error in our prediction.
# We will compare the results of various methods at the end.

#helper function to plot the stock prediction
prev_values = stock_price.iloc[:180]
y_test = stock_price.iloc[180:]

def plot_pred(pred,title):
    plt.figure(figsize=(17,5))
    plt.plot(prev_values,label='Train')
    plt.plot(y_test,label='Actual')
    plt.plot(pred,label='Predicted')
    plt.ylabel("Stock prices")
    plt.grid()
    plt.title(title,fontsize=20)
    plt.legend()
    plt.show()


# 1. Average ######################################################################

# This is the simplest model. We will get as average of the previous values and predict it as the forecast.

#Average of previous values
y_av = pd.Series(np.repeat(prev_values.mean(),68),index=y_test.index)
mse_score_avg = mse(y_av,y_test)

print(mse_score_avg)
print("\n")

plot_pred(y_av,"Average")

# 2. Weighted Mean ############################################

# We shall give more weightage to the data which are close to the last day in training data, while calculating the mean.
# The last day in the training set will get a weightage of 1(=180/180) and the first day will get a weightage of 1/180.

weight = np.array(range(0,180))/180
weighted_train_data =np.multiply(prev_values,weight)

# weighted average is the sum of this weighted train data by the sum of the weight

weighted_average = sum(weighted_train_data)/sum(weight)
y_wa = pd.Series(np.repeat(weighted_average,68),index=y_test.index)

mse_score_wa = mse(y_wa,y_test)

# coding is not something we see and understand but we do and understand. so practice it, try new things.
#print(weight)
#print(weighted_train_data)


#print(np.array(range(0,180)))

print(mse_score_wa)

plot_pred(y_wa,"Weighted Average")

# For the other methods we will predict the value of stock price on a day based on the values of
# stock prices of 80 days prior to it. So in our series we will not consider the first eight days
# (since there previous eighty days is not in the series).
# We have to test the last 68 values. This would be based on the last 80 days stock prices of each day in the test data.
# Since we have neglected first 80 and last 68 is our test set, the train dataset will be between 80 and 180 (100 days).

y_train = stock_price[80:180]
y_test = stock_price[180:]
print("y train:",y_train.shape,"\ny test:",y_test.shape)

# There are 100 days in training and 68 days in testing set. We will construct the features,
# that is the last 80 days stock for each date in the y_train and y_test. This would be our target variable.

X_train = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100)],
                       columns=range(80,0,-1),index=y_train.index)
X_test = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100,168)],
                       columns=range(80,0,-1),index=y_test.index)

print(X_train)
print("\n")

print(X_test)
print("\n")

#   X_train is now a collection of 100 dates as index and a collection of stock prices of previous 80 days as features.
#
#   Similarlily, X_test is now a collection of 68 dates as index and a collection of stock prices of previous 80 days as features.
#
#   NOTE: Here 76 working days from '2015-05-04', the stock had a price of 986.725 and 77 working days from '2015-05-05', the stock has the same value. You can see the similarity of values along the diagonal. This is because consecutitive data will be similar to the previous except it drops the last value, shifts and has a new value.
#
#   We will use these values for stock price prediction in the other four methods.


#print(X_test.mean(axis=1))

#print(y_test)

y_ma = X_test.mean(axis=1)
mse_score_ma = mse(y_ma,y_test)

print(mse_score_ma)

plot_pred(y_ma,"Moving Average")

# 4. Weighted Moving Average #####################################################################################

# We will obtain the stock price on the test date by calculating the weighted mean of past 80 days.
# The last of the 80 day will have a weightage of 1(=80/80) and the first will have a weightage of 1/80.

weight = np.array(range(1,81))/80


#weighted moving average
y_wma = X_test@weight/sum(weight)
mse_score_wma = mse(y_wma,y_test)

print(mse_score_wma)

plot_pred(y_wma,"Weighted Moving Average")

# 4. Linear regression ##############################################################################################
# In this method, we will perform a linear regression on our dataset.
# The values will be predicted as a linear combination of the previous 80 days values.

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
y_lr = pd.Series(y_lr,index=y_test.index)

mse_score_lr = mse(y_test,y_lr)

print(mse_score_lr)

plot_pred(y_lr,"Linear Regression")

# 6. Weighted Linear Regression ##########################################################################
# We will provide weightage to our input data rather than the features.

weight = np.array(range(1,101))/100
wlr = LinearRegression()

wlr.fit(X_train,y_train,weight)
y_wlr = wlr.predict(X_test)
y_wlr = pd.Series(y_wlr,index=y_test.index)

mse_score_wlr = mse(y_test,y_wlr)

print(mse_score_wlr)

plot_pred(y_wlr,"Weighted Linear Regression")

# 7. Lasso Regression ############################################################################################
# Linear Regression with L1 regulations.

from sklearn.linear_model import Lasso
lasso = Lasso()

las = lasso.fit(X_train,y_train)
y_las = las.predict(X_test)
y_las = pd.Series(y_las,index = y_test.index)

mse_score_lasso = mse(y_las,y_test)

print(mse_score_lasso)

plot_pred(y_las,"Lasso Regression")



