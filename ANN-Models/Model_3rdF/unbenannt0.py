# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:25:40 2021

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

Time_2W = df_building_clean.get('Floor1W')[['Time']]
Time_2S = df_building_clean.get('Floor3')[['Time']]


df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:22]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[23:42]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'PIR_ALL',df_building_clean.get('Floor1W')[Floor1W_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'CO2_ALL',df_building_clean.get('Floor1W').loc[:,Floor1W_heads[28:53]].sum(axis=1), True)


#get Input and test data
input_data1 = df_building_clean.get('Floor3')[['AP_Total']]
output_data1 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input1 = df_building_clean.get('Floor3')[['AP_Total']]
test_data_output1 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data2 = df_building_clean.get('Floor3')[['Inst_kW_Load_Plug','AP_Total']]
output_data2 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input2 = df_building_clean.get('Floor3')[['Inst_kW_Load_Plug','AP_Total']]
test_data_output2 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data3 = df_building_clean.get('Floor3')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
output_data3 = df_building_clean.get('Floor3')[['Groundtruth']]
test_data_input3 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
test_data_output3 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data = df_building_clean.get('Floor3')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL','PIR_ALL']]
output_data = df_building_clean.get('Floor3')[['Groundtruth']]
test_data_input = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL','PIR_ALL']]
test_data_output = df_building_clean.get('Floor1W')[['Groundtruth']]

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=99, test_size=0.2)
    
#get input and test data transformations    
input_scaler = preprocessing.StandardScaler().fit(x_train)
output_scaler = preprocessing.StandardScaler().fit(y_train)
    

def get_keras_model(h_layers,h_neurones, activation, m, n):
    inputs = keras.Input(shape=x_train.shape[1], name='input')

    layer_output = [inputs]
   
    for n_l in range(h_layers):

        layer_output.append(keras.layers.Dense(h_neurones,activation=activation,name='hidden_{}_relu'.format(n_l))(layer_output[-1]))
        
        layer_output.append(keras.layers.Dropout(n)(layer_output[n_l-1]))
        
    layer_output.append(keras.layers.Dense(y_train.shape[1], name='output')(layer_output[-1]))
    
    model = tf.keras.Model(inputs=inputs, outputs=layer_output[-1])
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=m),
    loss=tf.keras.losses.MSE
    )
        
    return model

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=50, verbose=0, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)
                

gt_model = get_keras_model(1,70,'relu',0.01,0.2)
history = gt_model.fit(
    input_scaler.transform(x_train),
    output_scaler.transform(y_train),
    batch_size = 200,
    epochs = 100,
    callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test), output_scaler.transform(y_test))
    )

gt_model.summary()
acc = history.history['loss']
val_loss = history.history['val_loss']

#gt_model.save("Model_1stF_allInputs2")


y_hat_scaled = gt_model.predict(input_scaler.transform(input_data))
y_hat = output_scaler.inverse_transform(y_hat_scaled)
w_1 = sklearn.metrics.r2_score(y_hat, output_data)

y_hat_scaled2 = gt_model.predict(input_scaler.transform(test_data_input))
y_hat2 = output_scaler.inverse_transform(y_hat_scaled2)
w_2 = sklearn.metrics.r2_score(y_hat2, test_data_output)

#plotting results of validation     
plt.figure(figsize=(12,5))
plt.title('Model 3rd-Floor [R_2:%s] ' %w_1, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2S, output_data,'ro',markersize=3,label='Groundtruth')     
plt.plot(Time_2S, y_hat,'b',label='prediction') 
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

        
plt.figure(figsize=(12,5))
plt.title('Model 3rd-Floor vs. Data 1st-FloorW, [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2W, y_hat2,'bo',markersize=5,label='Prediction')
plt.plot(Time_2W, test_data_output,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)