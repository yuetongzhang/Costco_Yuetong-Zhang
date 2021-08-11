import os
import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

monthly_sales = pd.read_csv(r'C:\Users\ZhangQ\Desktop\Costco_Monthly Sales from 2012 to 2021.csv')
monthly_sales1=monthly_sales.drop(columns=['Growth Rate'])
monthly_sales1=monthly_sales1.dropna()
monthly_sales_df=pd.DataFrame(monthly_sales1)
print(monthly_sales_df)

data_2020 = monthly_sales_df.loc[monthly_sales_df['Year'] == 2020]
Y_2020 = data_2020['Net Sales (billions)']
X_2020 = data_2020[['Month','Year']]

data_before = monthly_sales_df.loc[monthly_sales_df['Year'] < 2020]
Y_before = data_before['Net Sales (billions)']
X_before = data_before[['Month','Year']]

XTrain, XTest, YTrain, YTest = train_test_split(X_before, Y_before, test_size = 0.2, shuffle = True)
model = Sequential()
model.add(Dense(450, activation = 'relu', input_dim=2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(XTrain, YTrain, epochs=100, batch_size=128)

loss = model.evaluate(XTest, YTest)
print(np.sqrt(loss))
print(monthly_sales_df['Net Sales (billions)'].std())
predict_2020 = model.predict(X_2020)
r2_2020 = r2_score(Y_2020, predict_2020)

data_2021 = monthly_sales_df.loc[monthly_sales_df['Year'] == 2021]
Y_2021 = data_2021['Net Sales (billions)']
X_2021 = data_2021[['Month','Year']]

data_before1 = monthly_sales_df.loc[monthly_sales_df['Year'] < 2021]
Y_before1 = data_before1['Net Sales (billions)']
X_before1 = data_before1[['Month','Year']]

XTrain, XTest, YTrain, YTest = train_test_split(X_before1, Y_before1, test_size = 0.2, shuffle = True)
model = Sequential()
model.add(Dense(450, activation = 'relu', input_dim=2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(XTrain, YTrain, epochs=100, batch_size=128)

loss = model.evaluate(XTest, YTest)
print(np.sqrt(loss))
print(monthly_sales_df['Net Sales (billions)'].std())
predict_2021 = model.predict(X_2021)
r2_2021 = r2_score(Y_2021, predict_2021)

monthly_sales_future = pd.read_csv(r'C:\Users\ZhangQ\Desktop\Costco_Monthly Sales from 2021 to 2030.csv',index_col=0)
monthly_sales_future_df=pd.DataFrame(monthly_sales_future)
df_newDates=monthly_sales_future_df
Predicted_sales = model.predict(df_newDates)
new_dates_series=pd.Series(df_newDates.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)
new_predictions_series.to_csv("Costco_predicted_sales.csv",header=False)