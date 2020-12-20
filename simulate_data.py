import numpy as np
from scipy.io import loadmat
import scipy


N = 150 #number of samples
m=35# number of nodes (brain RIOs)
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=5):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
X1 = get_truncated_normal(mean=2, sd=1, low=1, upp=10)
X2 = get_truncated_normal(mean=8, sd=1, low=1, upp=10)
data1=X1.rvs([N,m,m])
data2=X2.rvs([N,m,m])
for i in range(N):

    data1[i, :, :] = data1[i, :, :] - np.diag(np.diag(data1[i,:,:])) #Removing diagonal elements of each matrixes
    data1[i, :,:] = (data1[i, :,:] + data1[i, :,:].transpose())/2 #Converting each matrixes symetric connectivity matrixes

    data2[i, :, :] = data2[i, :, :] - np.diag(np.diag(data2[i,:,:])) #Removing diagonal elements of each matrixes
    data2[i, :,:] = (data2[i, :,:] + data2[i, :,:].transpose())/2 #Converting each matrixes symetric connectivity matrixes

print("ok")
np.save('graph_view1', data1)#  shape (150,35,35)
np.save('graph_view2', data2)#  shape (150,35,35)
print("ok")