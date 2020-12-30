"""
To run: python3 main.py

"""
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



# returns adjacency matrix
def guassianSimilarity(X,sigma,k):
    A = np.zeros((len(X),len(X)))
    for i,x in enumerate(X):
        for j,y in enumerate(X):
            A[i][j] = np.exp(-1*(np.linalg.norm(x-y)**2))/(2*(sigma**2))
            if i==j:
                A[i][j] =0

#  only k most similar elements are considered               
    for i in range(len(A)):
        idx = np.argsort(-A[i])
        for j in range(len(A[0])):
            if idx[j]>k :
                A[i][j]=0

# make graph undirected graph    
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j]:
                A[j][i]=A[i][j]

    return A

# create the data
df = pd.read_csv('Dataset.csv')
X = df.to_numpy()
X =  preprocessing.normalize(X)
sigma = 1
A = guassianSimilarity(X,sigma,k=5)

 
# create the graph laplacian
D = np.diag(A.sum(axis=1))    
D_sqrt_inv = np.sqrt(np.linalg.inv(D))
L = np.matmul(np.matmul(D_sqrt_inv,D-A),D_sqrt_inv)


# find the eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)
# sort
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]
print('Eigenvalues :',vals)

# plot of eigen values
plt.ylabel('EigenValues') 
plt.xlabel('K-values')
plt.plot(np.array(vals).tolist(), linestyle='--', marker='o', color='b') 

#Find out how many clusters are there
# index of maximum difference in eigen value 
# suggest number of clusters
cluster=-1
maxdiff=0
for i in range(1,len(vals)-1):
    if vals[i+1]-vals[i]>maxdiff:
        maxdiff= vals[i+1]-vals[i]
        cluster = i+1
print('No of clusters :',cluster)

# k-mean clustering of reduced eigen vector sample space
kmeans = KMeans(n_clusters=cluster)
kmeans.fit(vecs[:,1:cluster])
colors = kmeans.labels_
print("Clusters:", colors)

#counting of samples in each clusters
cls = {} 
for i in colors:
    if i in cls.keys():
        cls[i]+=1
    else :
        cls[i]=1
print('Number of samples in ',cluster,' clusters are:' ,cls)



# For Data set summary 
# Y = df.to_numpy()
# print(np.mean(Y,axis=0))
# print(np.var(Y,axis=0))
# print(np.std(Y,axis=0))

