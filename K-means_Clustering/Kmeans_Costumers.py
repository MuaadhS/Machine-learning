
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('segmented_customers.csv')

df.info()
print('\n\n', df.head())

#check for null values
df.isnull().sum()
#df.dropna()

#show distributions
plt.figure(figsize = (15, 5))
plt.subplot(1,3,1)
plt.hist(df['Age'], bins = 15)
plt.title('Distribution of Age')
plt.subplot(1,3,2)
plt.hist(df['Annual Income (k$)'], bins = 15)
plt.title('Distribution of Annual Income')
plt.subplot(1,3,3)
plt.hist(df['Spending Score (1-100)'], bins = 15)
plt.title('Distribution of Spending Score')
plt.show()

#scatter 
plt.figure(figsize = (15 , 7))
plt.title('Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.scatter( x = 'Age', y = 'Spending Score (1-100)', data = df, s = 200)
plt.show()


from sklearn.cluster import KMeans

## First we cluster Age

#number of clusters
X1 = df[['Age', 'Spending Score (1-100)']].values
inertia = []

for n in range(1 , 15):
    kmeans = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 15) , inertia , 'o')
plt.plot(np.arange(1 , 15) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# k=4 seems like good option
kmeans = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

kmeans.fit(X1)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age', y = 'Spending Score (1-100)', data = df, c = labels, s = 100)
plt.scatter(x = centroids[: , 0] , y =  centroids[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()

## Second we cluster Annual Income

#number of clusters
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
inertia = []

for n in range(1 , 15):
    kmeans = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 15) , inertia , 'o')
plt.plot(np.arange(1 , 15) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# k=5
kmeans = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

kmeans.fit(X2)
labels2 = kmeans.labels_
centroids2 = kmeans.cluster_centers_

h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 
            s = 100 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()


# Combine Age and Annual Income
X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].values
inertia = []
for n in range(1 , 11):
    kmeans = (KMeans(n_clusters = n, init='k-means++', n_init = 10, max_iter=300, 
                        tol=0.0001, random_state= 111, algorithm='elkan'))
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

# k=6
kmeans = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

kmeans.fit(X3)
labels3 = kmeans.labels_
centroids3 = kmeans.cluster_centers_

y_kmeans = kmeans.fit_predict(X3)
df['cluster'] = pd.DataFrame(y_kmeans)
df.head()
df.to_csv("segmented_customers.csv", index = False)