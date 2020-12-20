"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import csv
import numpy
from sklearn import preprocessing
import urllib
import tensorflow as tf

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

import matplotlib.pyplot as plt
import networkx as nx
#import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
def load_data(path): #,type_,size,dataset):



    reg=np.load('vtrain1.npy')
    mal=np.load('vtrain2.npy')

    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
       data[i,:,:,0]=reg[i]
       data[i,:,:,1]=mal[i]

       return data




def load_data_test(size,dataset):

    reg=np.load('vtest1.npy')
    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
      data[i,:,:,0]=reg[i]
      data[i,:,:,1]=reg[i]
    return data
#  topological strength function
def degre_tf(tensor_obj):
    deg=tf.reduce_sum(tensor_obj,1)
    print('deg is shape',deg)
    return deg
