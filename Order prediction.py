# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:55:31 2022

@author: Muqeet
"""
#importing immportant modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/supplement.csv")
data.head()

#LOKING INto some of its necessery insights
data.info()

data.isnull().sum()

data.describe()

#explore important features

#pie plot
import plotly.express as px
pie = data["Store_Type"].value_counts()
store = pie.index
orders = pie.values
fig = px.pie(data, values=orders, names=store)
fig.show()

#pie plot
pie2 = data["Location_Type"].value_counts()
location = pie2.index
orders = pie2.values
fig = px.pie(data, values=orders, names=location)
fig.show()

#pie plot
pie3 = data["Discount"].value_counts()
discount = pie3.index
orders = pie3.values
fig = px.pie(data, values=orders, names=discount)
fig.show()

#pie plot
pie4 = data["Holiday"].value_counts()
holiday = pie4.index
orders = pie4.values
fig = px.pie(data, values=orders, names=holiday)
fig.show()



#lets prepare data so that we can train ML model

data["Discount"] = data["Discount"].map({"No": 0, "Yes": 1})
data["Store_Type"] = data["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
data["Location_Type"] = data["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})
data.dropna()


X=np.array(data[["Store_Type", "Location_Type","Holiday","Discount"]])
y = np.array(data["#Order"])

#splitting the data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=42,test_size=0.2)


import lightgbm as ltb
model = ltb.LGBMRegressor()
model.fit(xtrain, ytrain)

ypred=model.predict(xtest)
data = pd.DataFrame(data={"Predicted Orders": ypred.flatten()})
print(data.head())

