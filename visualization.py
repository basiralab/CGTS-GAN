import numpy as np
import pandas as pd
import seaborn as sns
#matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
#visualize input
gt=np.load('graph_view2.npy')  # ground truth data (foward translation eg view2)
sc=np.load('graph_view1.npy') # source data (backward translation eg view1)
n= 150 #number of samples in population.
m=35  #number of brain ROIs
gt=gt.reshape(-1)
sc=sc.reshape(-1)
a=sns.distplot(gt,hist=False,label='target view',color='Green')
b=sns.distplot(sc,hist=False,label='source view',color='Blue')
plt.gca().set(title='Simulated data distribution of view1 (source) and view2 (target)', ylabel='Distribution', xlabel='Connectivity Weight')
plt.legend(bbox_to_anchor=(1.005, 0.8), loc=7, borderaxespad=1)
plt.show()


#  visualise output (foward direction)
pred=np.load('wholepredicted_atob.npy')  # ground truth data (foward translation eg view2)
c=sns.distplot(gt,hist=False,label='target view',color='Green')
d=sns.distplot(pred,hist=False,label='predicted view',color='Blue')
plt.gca().set(title=' view1 to view2 translation', ylabel='Distribution', xlabel='Connectivity Weight')
plt.legend(bbox_to_anchor=(1.005, 0.8), loc=4, borderaxespad=1)
plt.show()

#  visualise output (opposite direction)
pred=np.load('wholepredicted_btoa.npy')  # ground truth data (foward translation eg view2)
c=sns.distplot(sc,hist=False,label='target view',color='Green')
d=sns.distplot(pred,hist=False,label='predicted view',color='Blue')
plt.gca().set(title=' view2 to view1 translation', ylabel='Distribution', xlabel='Connectivity Weight')
plt.legend(bbox_to_anchor=(1.005, 0.8), loc=4, borderaxespad=1)
plt.show()

###  visualizing graph matrix
mat=np.load('wholepredicted_atob.npy')
mat=mat.reshape(n,m,m)
plt.matshow(mat[1]);
plt.colorbar()
plt.title('Graph Matrix plot for predicted view by CGTS', y=1.08)
plt.show()