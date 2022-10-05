import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logistic regression dataset-Social_Network_Ads.csv')

#print(df.head())

#plt.scatter(df.Age, df.Purchased, marker = 'x', color = 'red')

# check how many labels are zeros or ones

#sizes = df.Purchased.value_counts(sort = 1)
#plt.pie(sizes, autopct = '%1.1f%%')  

#Drop irrelevant data
df.drop(['User ID'], axis = 1, inplace = True )
#print(df.head())

#drop the raw if there is a missing value
df = df.dropna()

#Convert gender into numeric value
df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 2
#print(df.head())

#prepare the data (x,y)
y = df.Purchased.values
x = df.drop(['Purchased'], axis = 1)
#print(x.head())


#split the data into training and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.1, random_state=20)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#define the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

#testing
prediction_test = model.predict(x_test)

#check accuracy
from sklearn import metrics

print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

#check weights
weights = pd.Series(model.coef_[0], index = x.columns.values)
print(weights)
