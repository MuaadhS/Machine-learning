# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:57:38 2022

@author: moaadh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("cells.csv") 
print(df, "\n")

# series 
x = df.time
y = df.cells

#Alternative method
# x = df['time']

#Get entire column
# dataframe
x_df = df[['time']]

#alternative method:
# x_df = df.drop('cells', axis = 'columns')

y_df = df.drop('time', axis = 'columns')


plt.title('Scattered features')
plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(x, y, color = 'red', marker = 'x')
plt.show()

#print(x)
print(x_df, "\n")
print(y_df, "\n")

model = linear_model.LinearRegression()
model.fit(x_df.values, y_df.values)

print(model.score(x_df, y_df))
print("Weight (w) =", model.intercept_ )
print("Bias (b) = ", model.coef_)

plt.title('Linear regression line')
plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(x, y, color = 'red', marker = 'x')
plt.plot( np.sort(x_df, axis = 0), model.predict(x_df))
plt.show()

run = True
 
while run:
    user_input = input("Enter the time here (enter exit to leave): ")
    if user_input == "exit":
        run = False
        print("Sorry to see you go. Take care buddy!")
    else:
        print("Predict number of cells = ", model.predict([[user_input]]))