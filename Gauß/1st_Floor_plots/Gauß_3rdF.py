# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:10:13 2021

@author: Felix
"""
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

df_building_clean = {}
for k1 in df_building.keys():
    df_building_clean[k1] = df_building.get(k1).drop('Unnamed: 0',axis=1)
    
    
Floor1S_heads = list(df_building_clean.get('Floor1S').columns)
Floor1W_heads = list(df_building_clean.get('Floor1W').columns)
Floor2S_heads = list(df_building_clean.get('Floor2S').columns)
Floor2W_heads = list(df_building_clean.get('Floor2W').columns)
Floor3_heads = list(df_building_clean.get('Floor3').columns)
Floor4_heads = list(df_building_clean.get('Floor4').columns)

Time_2W = df_building_clean.get('Floor3')[['Time']]



X = df_building_clean.get('Floor3')[['AP_Total']].values
X = (np.atleast_2d(X))

y = df_building_clean.get('Floor3')[['Groundtruth']].iloc[:,0].values
y = y.reshape(-1,1)
x = (np.atleast_2d(np.linspace(1,100,289))).T

X2 = df_building_clean.get('Floor1W')[['AP_Total']].values
X2 = (np.atleast_2d(X2))
y2 = df_building_clean.get('Floor1W')[['Groundtruth']].iloc[:,0].values
y2 = y2.reshape(-1,1)
x2 = (np.atleast_2d(np.linspace(1,100,127))).T


# Kernel with parameters given in GPML book
k1 = 30.0**2 * RBF(length_scale=35)  # long term smooth rising trend

k4 = 4**2 * RBF(length_scale=1) \
    + WhiteKernel(noise_level=5**2)  # noise terms
kernel = k1 + k4

gp = GaussianProcessRegressor(kernel=kernel,optimizer='fmin_l_bfgs_b', n_restarts_optimizer=20,normalize_y=True)
gp.fit(X,y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)



plt.figure(figsize=(10,7))
plt.scatter(X, y, c='k',label='Data points')
plt.plot(X_, y_pred, 'g',label='Mean of predictive distribution')



y_pred = np.reshape(y_pred,-1)
plt.fill_between(X_[:, 0], (y_pred - y_std), (y_pred + y_std),
                  alpha=0.5, color='b',label='Standard deviation of predictive distribution')

plt.xlim(X_.min(), X_.max())

plt.xlabel("AP_Total")
plt.ylabel(r"Occupancy count")
plt.title(r"Gaussian process, 1st-Floor")
plt.legend(loc='lower center')
plt.show()

y_pred2, sigma = gp.predict(X, return_std=True)

w_2 = sklearn.metrics.r2_score(y_pred2, y)

plt.figure(figsize=(12,5))
plt.title('Model_gauss 3rd-FloorS vs. Data 3rd-FloorS, [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(x, y_pred2,'bo',markersize=5,label='Prediction')
plt.plot(x, y,'ro',markersize=5,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time_steps')
plt.ylabel('Occupancy-count')
plt.grid(True)




y_pred3, sigma2 = gp.predict(X2, return_std=True)

w_2 = sklearn.metrics.r2_score(y_pred3, y2)



plt.figure(figsize=(12,5))
plt.title('Model_gauss 3rd-FloorS vs. Data 1st-FloorW, [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(x2, y_pred3,'bo',markersize=5,label='Prediction')
plt.plot(x2, y2,'ro',markersize=5,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time_steps')
plt.ylabel('Occupancy-count')
plt.grid(True)