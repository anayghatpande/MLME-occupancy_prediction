# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:23:14 2021

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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
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
Time_2S = df_building_clean.get('Floor4')[['Time']]
Time2S = df_building_clean.get('Floor2S')[['Time']]
Time1S = df_building_clean.get('Floor1S')[['Time']]
Time1W = df_building_clean.get('Floor1W')[['Time']]
Time2W = df_building_clean.get('Floor2W')[['Time']]




df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[28:53]].sum(axis=1), True)
df_building_clean.get('Floor4').insert(0, 'PIR_ALL',df_building_clean.get('Floor4').loc[:,Floor4_heads[1:20]].sum(axis=1), True)
df_building_clean.get('Floor4').insert(1, 'CO2_ALL',df_building_clean.get('Floor4').loc[:,Floor4_heads[20:36]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(0, 'PIR_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(1, 'CO2_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(0, 'PIR_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(1, 'CO2_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)





#get Input and test data
input_data2S = df_building_clean.get('Floor2S')[['Inst_kW_Load_Elec','AP_Total']].values
output_data2S = df_building_clean.get('Floor2S')[['Groundtruth']].values

input_data2W = df_building_clean.get('Floor2W')[['Inst_kW_Load_Elec','AP_Total']].values
output_data2W = df_building_clean.get('Floor2W')[['Groundtruth']].values

input_data1W = df_building_clean.get('Floor1W')[['Inst_kW_Load_Elec','AP_Total']].values
output_data1W = df_building_clean.get('Floor1W')[['Groundtruth']].values

input_data1S = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total']].values
output_data1S = df_building_clean.get('Floor1S')[['Groundtruth']].values

Train_data = np.vstack((input_data2W,input_data1W,input_data1S))
Train_output = np.vstack((output_data2W,output_data1W,output_data1S))

#Test set which the ANN has never seen before
test_data_input = df_building_clean.get('Floor3')[['Inst_kW_Load_Elec','AP_Total']]
test_data_output = df_building_clean.get('Floor3')[['Groundtruth']]

#The following is just to recreate the scaling factors
input_data = df_building_clean.get('Floor4')[['Inst_kW_Load_Elec','AP_Total']]
output_data = df_building_clean.get('Floor4')[['Groundtruth']]

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=99, test_size=0.2)

input_scaler = preprocessing.StandardScaler().fit(x_train)
output_scaler = preprocessing.StandardScaler().fit(y_train)


#Train the model
X_train, X_test, Y_train, Y_test = train_test_split(Train_data, Train_output, random_state=99, test_size=0.2)

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=70, verbose=0, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

model = keras.models.load_model('4th-Floor_Model')

history = model.fit(
    input_scaler.transform(X_train),
    output_scaler.transform(Y_train),
    batch_size = 20,
    epochs = 400,
    callbacks=[es,rlr],
    validation_data=(input_scaler.transform(X_test), output_scaler.transform(Y_test))
    )

model.summary()
acc = history.history['loss']
val_loss = history.history['val_loss']

# model.save('Model-universal')


#Test the model on all datasets
y_hat_scaled = model.predict(input_scaler.transform(input_data))
y_hat = output_scaler.inverse_transform(y_hat_scaled)
w_1 = sklearn.metrics.r2_score(y_hat, output_data)

y_hat_scaled2 = model.predict(input_scaler.transform(test_data_input))
y_hat2 = output_scaler.inverse_transform(y_hat_scaled2)
w_2 = sklearn.metrics.r2_score(y_hat2, test_data_output)

y_hat_scaled3 = model.predict(input_scaler.transform(input_data2S))
y_hat3 = output_scaler.inverse_transform(y_hat_scaled3)
w_3 = sklearn.metrics.r2_score(y_hat3, output_data2S)

y_hat_scaled4 = model.predict(input_scaler.transform(input_data1S))
y_hat4 = output_scaler.inverse_transform(y_hat_scaled4)
w_4 = sklearn.metrics.r2_score(y_hat4, output_data1S)

y_hat_scaled5 = model.predict(input_scaler.transform(input_data1W))
y_hat5 = output_scaler.inverse_transform(y_hat_scaled5)
w_5 = sklearn.metrics.r2_score(y_hat5, output_data1W)

y_hat_scaled6 = model.predict(input_scaler.transform(input_data2W))
y_hat6 = output_scaler.inverse_transform(y_hat_scaled6)
w_6 = sklearn.metrics.r2_score(y_hat6, output_data2W)

#plotting results of validation     
plt.figure(figsize=(12,5))
plt.title('Model vs. Data 4th-Floor [R_2:%s] ' %w_1, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2S, y_hat,'b',markersize=5,label='Prediction')
plt.plot(Time_2S, output_data,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)


        
plt.figure(figsize=(12,5))
plt.title('Model vs. Data 3rd-FloorW, [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2W, y_hat2,'bo',markersize=5,label='Prediction')
plt.plot(Time_2W, test_data_output,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

plt.figure(figsize=(12,5))
plt.title('Model vs. Data 2nd-FloorS, [R_2:%s] ' %w_3, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time2S, y_hat3,'bo',markersize=5,label='Prediction')
plt.plot(Time2S, output_data2S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

plt.figure(figsize=(12,5))
plt.title('Model vs. Data 1st-FloorS, [R_2:%s] ' %w_4, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time1S, y_hat4,'bo',markersize=5,label='Prediction')
plt.plot(Time1S, output_data1S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

plt.figure(figsize=(12,5))
plt.title('Model vs. Data 1st-FloorW, [R_2:%s] ' %w_5, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time1W, y_hat5,'bo',markersize=5,label='Prediction')
plt.plot(Time1W, output_data1W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

plt.figure(figsize=(12,5))
plt.title('Model vs. Data 2nd-FloorW, [R_2:%s] ' %w_6, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time2W, y_hat6,'bo',markersize=5,label='Prediction')
plt.plot(Time2W, output_data2W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)
