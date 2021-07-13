# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:19:23 2021

@author: Felix
"""
#I always copyed the same set of libraries, a lot of them are unsused

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.layers import Dropout, Dense
from datetime import datetime
from sklearn.gaussian_process.kernels \
import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.gaussian_process.kernels import ConstantKernel as C

from sklearn.gaussian_process import GaussianProcessRegressor


df_building = pd.read_excel('Clean_Set.xlsx',sheet_name = None)

#Here i drop an unused column in every df in the dict.

df_building_clean = {}
for k1 in df_building.keys():
    df_building_clean[k1] = df_building.get(k1).drop('Unnamed: 0',axis=1)
    

#Here I save the heads of the df's in lis    
Floor1S_heads = list(df_building_clean.get('Floor1S').columns)
Floor1W_heads = list(df_building_clean.get('Floor1W').columns)
Floor2S_heads = list(df_building_clean.get('Floor2S').columns)
Floor2W_heads = list(df_building_clean.get('Floor2W').columns)
Floor3_heads = list(df_building_clean.get('Floor3').columns)
Floor4_heads = list(df_building_clean.get('Floor4').columns)


#Time of df for plotting ( must be changed when using the code on different floor)
Time_2W = df_building_clean.get('Floor1S')[['Time']]

#Summing up PIR and CO2. Here the lists of heads are useful to choose the correct heads
df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[28:53]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'PIR_ALL',df_building_clean.get('Floor1W')[Floor1W_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'CO2_ALL',df_building_clean.get('Floor1W').loc[:,Floor1W_heads[28:53]].sum(axis=1), True)

#here the data to fit the GP is defined. To change the input features I just changed the corresponding variable from X1[1-3] to X
X11 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL','CO2_ALL']].values #alpha=0.1
X12 = df_building_clean.get('Floor1S')[['AP_Total','Inst_kW_Load_Plug','PIR_ALL']].values #no alpha defined
X13 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total']].values #no alpha defined

X = df_building_clean.get('Floor1S')[['AP_Total']].values #no alpha defined
X = (np.atleast_2d(X))


y = df_building_clean.get('Floor1S')[['Groundtruth']].iloc[:,0].values
y = y.reshape(-1,1)
x = (np.atleast_2d(np.linspace(1,100,len(Time_2W)))).T


#Doing the same for a test data set
X21 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL','CO2_ALL']].values
X22 = df_building_clean.get('Floor1W')[['AP_Total','Inst_kW_Load_Plug','PIR_ALL']].values
X23 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total']].values
X2 = df_building_clean.get('Floor1W')[['AP_Total']].values
X2 = (np.atleast_2d(X2))
y2 = df_building_clean.get('Floor1W')[['Groundtruth']].iloc[:,0].values
y2 = y2.reshape(-1,1)
x2 = (np.atleast_2d(np.linspace(1,100,127))).T

#defining the scaling factors
input_scaler = preprocessing.StandardScaler().fit(X)
output_scaler = preprocessing.StandardScaler().fit(y)


#Defining Kernel functions
k1 = 1**2 * RBF(length_scale=1)  # long term smooth rising trend


k4 = 1**2 * RBF(length_scale=1) \
    + WhiteKernel(noise_level=0.2**2)  # noise terms
    
kernel = k1 + k4



#Fitting GP, alpha here turns out to be the most inportent parameterto avoid overfitting
gp = GaussianProcessRegressor(kernel=kernel,optimizer='fmin_l_bfgs_b', n_restarts_optimizer=20,normalize_y=False)#,alpha=0.1)
gp.fit(input_scaler.transform(X), output_scaler.transform(y))

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))


#Testing the fitted model
y_pred, sigma = gp.predict(input_scaler.transform(X), return_std=True)
y_hat = output_scaler.inverse_transform(y_pred)
w_2 = sklearn.metrics.r2_score(y_hat, y)

plt.figure(figsize=(12,5))
plt.title('Model_gauss 1st-FloorS vs. Data 1st-FloorS (AP+kW+PIR+C), [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(x, y_hat,'bo',markersize=5,label='Prediction')
plt.plot(x, y,'ro',markersize=5,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time_steps')
plt.ylabel('Occupancy-count')
plt.grid(True)




y_pred2, sigma2 = gp.predict(input_scaler.transform(X2), return_std=True)
y_hat2 = output_scaler.inverse_transform(y_pred2)
w_2 = sklearn.metrics.r2_score(y_hat2, y2)

plt.figure(figsize=(12,5))
plt.title('Model_gauss 1st-FloorS vs. Data 1st-FloorW (AP+kW+PIR+C), [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(x2, y_hat2,'bo',markersize=5,label='Prediction')
plt.plot(x2, y2,'ro',markersize=5,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time_steps')
plt.ylabel('Occupancy-count')
plt.grid(True)