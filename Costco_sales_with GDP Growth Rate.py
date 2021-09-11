#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import numpy as np
import pandas as pd


# In[20]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[21]:


monthly_sales = pd.read_csv('https://raw.githubusercontent.com/yuetongzhang/Costco_Yuetong-Zhang/main/Costco_Yuetong%20Zhang/Costco_Monthly%20Sales%20from%202012%20to%202021.csv')
monthly_sales1=monthly_sales.drop(columns=['Growth Rate','Date'])
monthly_sales1=monthly_sales1.dropna()
monthly_sales_df=pd.DataFrame(monthly_sales1)
monthly_sales_df = monthly_sales_df.rename(columns={'Net Sales (billions)': 'Total Sales'})
monthly_sales_df = monthly_sales_df[['Total Sales', 'Year', 'Month']]
#print(monthly_sales_df)
monthly_sales_df.head()


# In[22]:


real_rate=pd.read_csv('https://raw.githubusercontent.com/RamenMode/NeuralNetsSales/main/100-EmploymentRateReal.csv')
df_real_rate=pd.DataFrame(real_rate)
df_real_rate.head()
irate= df_real_rate.iloc[:,3]
monthly_sales_df['GDP Growth Rate']=irate
print(monthly_sales_df)


# In[23]:


len(monthly_sales_df)


# In[24]:


data_2020 = monthly_sales_df.loc[monthly_sales_df['Year'] == 2020]
Y_2020 = data_2020['Total Sales']
X_2020 = data_2020[['Month','Year','GDP Growth Rate']]


# In[25]:


data_before = monthly_sales_df.loc[monthly_sales_df['Year'] < 2020]
Y_before = data_before['Total Sales']
X_before = data_before[['Month','Year','GDP Growth Rate']]


# In[26]:


XTrain, XTest, YTrain, YTest = train_test_split(X_before, Y_before, test_size = 0.2, shuffle = True)
#XTrain = X_before
#YTrain = Y_before

model0 = Sequential()
model0.add(Dense(450, activation = 'relu', input_dim=3))
model0.add(Dense(200, activation = 'relu'))
model0.add(Dropout(0.1))
model0.add(Dense(100, activation = 'relu'))
model0.add(Dropout(0.2))
model0.add(Dense(1, activation = 'linear'))
model0.compile(optimizer='adam', loss='mean_squared_error')


# In[27]:


loss30=0
loss2020=0
for i in range(30):
  model0.fit(XTrain, YTrain, epochs=100, batch_size=128)
  loss1 = model0.evaluate(XTest, YTest)
  loss2 = model0.evaluate(X_2020, Y_2020)
  loss30+=loss1
  loss2020+=loss2
  
  
  


# In[28]:


print(np.sqrt(loss30/30))


# In[29]:


#loss = model0.evaluate(XTest, YTest)
print(np.sqrt(loss30/30))
print(monthly_sales_df['Total Sales'].std())
predict_2020 = model0.predict(X_2020)
r2_2020 = r2_score(Y_2020, predict_2020)


# In[30]:


XTest = X_2020
YTest = Y_2020
#loss2020 = model0.evaluate(XTest, YTest)
print(np.sqrt(loss2020/30))
print(monthly_sales_df['Total Sales'].std())


# In[31]:


data_2021 = monthly_sales_df.loc[monthly_sales_df['Year'] == 2021]
Y_2021 = data_2021['Total Sales']
X_2021 = data_2021[['Month','Year']]


# In[ ]:




