# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:59:57 2021

@author: Kai
"""


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

# gt_model = keras.models.load_model("Model_3rdF_allInputs2_1st_FloorS")

#Getting the data

df_building = pd.read_excel('Clean_set.xlsx',sheet_name = None)

df_building_clean = {}
for k1 in df_building.keys():
    df_building_clean[k1] = df_building.get(k1).drop('Unnamed: 0',axis=1)
    
Floor1S_heads = list(df_building_clean.get('Floor1S').columns)
Floor1W_heads = list(df_building_clean.get('Floor1W').columns)
Floor2S_heads = list(df_building_clean.get('Floor2S').columns)
Floor2W_heads = list(df_building_clean.get('Floor2W').columns)
Floor3_heads = list(df_building_clean.get('Floor3').columns)
Floor4_heads = list(df_building_clean.get('Floor4').columns)

#Putting the data in the right configuration

df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(2, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[27:53]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'PIR_ALL',df_building_clean.get('Floor1W')[Floor1W_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(2, 'CO2_ALL',df_building_clean.get('Floor1W').loc[:,Floor1W_heads[27:53]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(1, 'PIR_ALL',df_building_clean.get('Floor2S')[Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(2, 'CO2_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[11:24]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(1, 'PIR_ALL',df_building_clean.get('Floor2W')[Floor2W_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(2, 'CO2_ALL',df_building_clean.get('Floor2W').loc[:,Floor2W_heads[11:24]].sum(axis=1), True)
df_building_clean.get('Floor3').insert(1, 'PIR_ALL',df_building_clean.get('Floor3')[Floor3_heads[1:22]].sum(axis=1), True)
df_building_clean.get('Floor3').insert(2, 'CO2_ALL', df_building_clean.get('Floor3').loc[:,Floor3_heads[22:42]].sum(axis=1), True)
df_building_clean.get('Floor4').insert(1, 'PIR_ALL',df_building_clean.get('Floor4')[Floor4_heads[1:20]].sum(axis=1), True)
df_building_clean.get('Floor4').insert(2, 'CO2_ALL', df_building_clean.get('Floor4').loc[:,Floor4_heads[20:36]].sum(axis=1), True)

#New Zero for the AP sensors

df_building_clean.get('Floor1S').insert(3, 'AP_Total_Zero', df_building_clean.get('Floor1S')['AP_Total']-10.466)
df_building_clean.get('Floor2S').insert(3, 'AP_Total_Zero', df_building_clean.get('Floor2S')['AP_Total']-10.466)
df_building_clean.get('Floor3').insert(3, 'AP_Total_Zero', df_building_clean.get('Floor3')['AP_Total']-5.279)
df_building_clean.get('Floor4').insert(3, 'AP_Total_Zero', df_building_clean.get('Floor4')['AP_Total']-10.442)

#Define the input data and the test set

input_data  = df_building_clean.get('Floor4')[['AP_Total_Zero']]
output_data = df_building_clean.get('Floor4')[['Groundtruth']]

input_data_1_S  = df_building_clean.get('Floor1S')[['AP_Total_Zero']]
output_data_1_S = df_building_clean.get('Floor1S')[['Groundtruth']]

input_data_1_W  = df_building_clean.get('Floor1W')[['AP_Total']]
output_data_1_W = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data_2_S = df_building_clean.get('Floor2S')[['AP_Total_Zero']]
output_data_2_S = df_building_clean.get('Floor2S')[['Groundtruth']]

input_data_2_W  = df_building_clean.get('Floor2W')[['AP_Total']]
output_data_2_W = df_building_clean.get('Floor2W')[['Groundtruth']]

input_data_3   = df_building_clean.get('Floor3')[['AP_Total_Zero']]
output_data_3  = df_building_clean.get('Floor3')[['Groundtruth']]



test_data_input_1S  = df_building_clean.get('Floor1S')[['AP_Total_Zero']]
test_data_output_1S = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input_1W  = df_building_clean.get('Floor1W')[['AP_Total', ]]
test_data_output_1W = df_building_clean.get('Floor1W')[['Groundtruth']]
test_data_input_2S  = df_building_clean.get('Floor2S')[['AP_Total_Zero']]
test_data_output_2S = df_building_clean.get('Floor2S')[['Groundtruth']]
test_data_input_2W  = df_building_clean.get('Floor2W')[['AP_Total']]
test_data_output_2W = df_building_clean.get('Floor2W')[['Groundtruth']]
test_data_input_3   = df_building_clean.get('Floor3')[['AP_Total_Zero']]
test_data_output_3  = df_building_clean.get('Floor3')[['Groundtruth']]
test_data_input_4   = df_building_clean.get('Floor4')[['AP_Total_Zero']]
test_data_output_4  = df_building_clean.get('Floor4')[['Groundtruth']]

#Create the time line 

Time_1S = df_building_clean.get('Floor1S')[['Time']]
Time_1W = df_building_clean.get('Floor1W')[['Time']]
Time_2S = df_building_clean.get('Floor2S')[['Time']]
Time_2W = df_building_clean.get('Floor2W')[['Time']]
Time_3  = df_building_clean.get('Floor3')[['Time']]
Time_4  = df_building_clean.get('Floor4')[['Time']]


#Split training and validation set

x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(input_data, output_data, random_state=99, test_size=0.2)
    
#get input and test data transformations    
input_scaler = preprocessing.StandardScaler().fit(x_train_4)
output_scaler = preprocessing.StandardScaler().fit(y_train_4)

x_train_1_S, x_test_1_S, y_train_1_S, y_test_1_S = train_test_split(input_data_1_S, output_data_1_S, random_state=99, test_size=0.2)
x_train_1_W, x_test_1_W, y_train_1_W, y_test_1_W = train_test_split(input_data_1_W, output_data_1_W, random_state=99, test_size=0.2)
x_train_2_S, x_test_2_S, y_train_2_S, y_test_2_S = train_test_split(input_data_2_S, output_data_2_S, random_state=99, test_size=0.2)
x_train_2_W, x_test_2_W, y_train_2_W, y_test_2_W = train_test_split(input_data_2_W, output_data_2_W, random_state=99, test_size=0.2)
x_train_3, x_test_3, y_train_3, y_test_3         = train_test_split(input_data_3, output_data_3, random_state=99, test_size=0.2)

# #Create Keras Model

def get_keras_model(h_layers,h_neurones, activation, m, act_2):#, n):
    
    inputs = keras.Input(shape=x_train_4.shape[1], name='input')

    layer_output = [inputs]
   
    for n_l in range(h_layers):
        if n_l == h_layers-1:
            layer_output.append(keras.layers.Dense(h_neurones,activation= act_2,name='hidden_{}_sigmoid'.format(n_l))(layer_output[-1]))
        else:
            layer_output.append(keras.layers.Dense(h_neurones,activation=activation,name='hidden_{}_relu'.format(n_l))(layer_output[-1]))
        
        #layer_output.append(keras.layers.Dropout(n)(layer_output[n_l-1]))
        
    layer_output.append(keras.layers.Dense(y_train_4.shape[1], name='output')(layer_output[-1]))
    
    model = tf.keras.Model(inputs=inputs, outputs=layer_output[-1])
    model.compile(
        
    optimizer=tf.keras.optimizers.Adam(learning_rate=m),
    loss=tf.keras.losses.MSE
    )
        
    return model
#Define Callbacks for training

# # es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=20, verbose=0, restore_best_weights=False)
# # rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)


#Train the Model

gt_model = get_keras_model(1,81,'relu',0.05,'sigmoid')#,n)

history = gt_model.fit(
    input_scaler.transform(x_train_4),
    output_scaler.transform(y_train_4),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

# gt_model.summary()
# acc = history.history['loss']
# val_loss = history.history['val_loss']

#Train the model

history = gt_model.fit(
    input_scaler.transform(x_train_1_S),
    output_scaler.transform(y_train_1_S),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

# gt_model.summary()
# acc = history.history['loss']
# val_loss = history.history['val_loss']


history = gt_model.fit(
    input_scaler.transform(x_train_1_W),
    output_scaler.transform(y_train_1_W),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

# gt_model.summary()
# acc = history.history['loss']
# val_loss = history.history['val_loss']


history = gt_model.fit(
    input_scaler.transform(x_train_2_S),
    output_scaler.transform(y_train_2_S),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

# gt_model.summary()
# acc = history.history['loss']
# val_loss = history.history['val_loss']


history = gt_model.fit(
    input_scaler.transform(x_train_2_W),
    output_scaler.transform(y_train_2_W),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

# gt_model.summary()
# acc = history.history['loss']
# val_loss = history.history['val_loss']


history = gt_model.fit(
    input_scaler.transform(x_train_3),
    output_scaler.transform(y_train_3),
    batch_size = 30,
    epochs = 110,
    # callbacks = [es,rlr],
    validation_data=(input_scaler.transform(x_test_4), output_scaler.transform(y_test_4))
    )

gt_model.summary()
acc = history.history['loss']
val_loss = history.history['val_loss']


##############################################################################
gt_model.save("Model_4thF_1Inputs_1_81_0.05_30_110_relu_sigmoid_EP")
##############################################################################


# Evaluate the Model

y_hat_scaled = gt_model.predict(input_scaler.transform(x_test_4))
y_hat = output_scaler.inverse_transform(y_hat_scaled)
w_1 = sklearn.metrics.r2_score(y_hat, y_test_4)

y_hat_scaled_1S = gt_model.predict(input_scaler.transform(test_data_input_1S))
y_hat_1S = output_scaler.inverse_transform(y_hat_scaled_1S)
w_1S = sklearn.metrics.r2_score(y_hat_1S, test_data_output_1S)

y_hat_scaled_1W = gt_model.predict(input_scaler.transform(test_data_input_1W))
y_hat_1W = output_scaler.inverse_transform(y_hat_scaled_1W)
w_1W = sklearn.metrics.r2_score(y_hat_1W, test_data_output_1W)

y_hat_scaled_2S = gt_model.predict(input_scaler.transform(test_data_input_2S))
y_hat_2S = output_scaler.inverse_transform(y_hat_scaled_2S)
w_2S = sklearn.metrics.r2_score(y_hat_2S, test_data_output_2S)

y_hat_scaled_2W = gt_model.predict(input_scaler.transform(test_data_input_2W))
y_hat_2W = output_scaler.inverse_transform(y_hat_scaled_2W)
w_2W = sklearn.metrics.r2_score(y_hat_2W, test_data_output_2W)

y_hat_scaled_3 = gt_model.predict(input_scaler.transform(test_data_input_3))
y_hat_3 = output_scaler.inverse_transform(y_hat_scaled_3)
w_3 = sklearn.metrics.r2_score(y_hat_3, test_data_output_3)

y_hat_scaled_4 = gt_model.predict(input_scaler.transform(test_data_input_4))
y_hat_4 = output_scaler.inverse_transform(y_hat_scaled_4)
w_4 = sklearn.metrics.r2_score(y_hat_4, test_data_output_4)

#Plot the results

plt.figure(figsize=(3,5))
plt.title('Validation-Set vs. Prediction [R_2:%s] ' %w_1, fontsize=15, color= 'black', y= 1.1)
      
plt.plot(y_hat,y_test_4,'bo')
plt.xlabel('Occupancy-count [predicted]')
plt.ylabel('Occupancy-count [true]')       
plt.grid(True)


#4th Floor vs 1st Floor Summer        
plt.figure(figsize=(12,5))
plt.title('1st-Floor Summer, [R_2:%s] ' %w_1S, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_1S, y_hat_1S,'bo',markersize=5,label='Prediction')
plt.plot(Time_1S, test_data_output_1S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 1st Floor Winter  
plt.figure(figsize=(12,5))
plt.title('1st-Floor Winter, [R_2:%s] ' %w_1W, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_1W, y_hat_1W,'bo',markersize=5,label='Prediction')
plt.plot(Time_1W, test_data_output_1W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 2nd Floor Summer  
plt.figure(figsize=(12,5))
plt.title('2nd-Floor Summer, [R_2:%s] ' %w_2S, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2S, y_hat_2S,'bo',markersize=5,label='Prediction')
plt.plot(Time_2S, test_data_output_2S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 2nd Floor Winter  
plt.figure(figsize=(12,5))
plt.title('2nd-Floor Winter, [R_2:%s] ' %w_2W, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2W, y_hat_2W,'bo',markersize=5,label='Prediction')
plt.plot(Time_2W, test_data_output_2W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 3rd Floor  
plt.figure(figsize=(12,5))
plt.title('3rd-Floor, [R_2:%s] ' %w_3, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_3, y_hat_3,'bo',markersize=5,label='Prediction')
plt.plot(Time_3, test_data_output_3,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 4th Floor  
plt.figure(figsize=(12,5))
plt.title('4th-Floor, [R_2:%s] ' %w_4, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_4, y_hat_4,'bo',markersize=5,label='Prediction')
plt.plot(Time_4, test_data_output_4,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)


##################################################################################################
reconstructed_model = keras.models.load_model("Model_4thF_1Inputs_1_81_0.05_30_110_relu_sigmoid_EP")
##################################################################################################

#Evaluate the saved Model

y_hat_scaled_t = gt_model.predict(input_scaler.transform(input_data))
y_hat_t = output_scaler.inverse_transform(y_hat_scaled_t)
w_1_t = sklearn.metrics.r2_score(y_hat, y_test_4)

y_hat_scaled_1S_t = gt_model.predict(input_scaler.transform(test_data_input_1S))
y_hat_1S_t = output_scaler.inverse_transform(y_hat_scaled_1S_t)
w_1S_t = sklearn.metrics.r2_score(y_hat_1S_t, test_data_output_1S)

y_hat_scaled_1W_t = gt_model.predict(input_scaler.transform(test_data_input_1W))
y_hat_1W_t = output_scaler.inverse_transform(y_hat_scaled_1W_t)
w_1W_t = sklearn.metrics.r2_score(y_hat_1W_t, test_data_output_1W)

y_hat_scaled_2S_t = gt_model.predict(input_scaler.transform(test_data_input_2S))
y_hat_2S_t = output_scaler.inverse_transform(y_hat_scaled_2S_t)
w_2S_t = sklearn.metrics.r2_score(y_hat_2S, test_data_output_2S)

y_hat_scaled_2W_t = gt_model.predict(input_scaler.transform(test_data_input_2W))
y_hat_2W_t = output_scaler.inverse_transform(y_hat_scaled_2W_t)
w_2W_t = sklearn.metrics.r2_score(y_hat_2W_t, test_data_output_2W)

y_hat_scaled_3_t = gt_model.predict(input_scaler.transform(test_data_input_3))
y_hat_3_t = output_scaler.inverse_transform(y_hat_scaled_3_t)
w_3_t = sklearn.metrics.r2_score(y_hat_3_t, test_data_output_3)

y_hat_scaled_4 = gt_model.predict(input_scaler.transform(test_data_input_4))
y_hat_4 = output_scaler.inverse_transform(y_hat_scaled_4)
w_4 = sklearn.metrics.r2_score(y_hat_4, test_data_output_4)

#Plot the results again

# plt.figure(figsize=(12,5))
# plt.title('Model 4th-Floor vs. Data 4th-Floor, [R_2:%s] ' %w_1_t, fontsize=15, color= 'black', y= 1.1)
# plt.plot(Time_4, y_hat_t,'b',markersize=5,label='Prediction')
# plt.plot(Time_4, output_data,'ro',markersize=3,label='Groundtruth')
# plt.legend(loc='upper center')
# plt.xlabel('Time')
# plt.ylabel('Occupancy-count')
# plt.grid(True)


#4th Floor vs 1st Floor Summer        
plt.figure(figsize=(12,5))
plt.title('1st-Floor Summer, [R_2:%s] ' %w_1S_t, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_1S, y_hat_1S_t,'bo',markersize=5,label='Prediction')
plt.plot(Time_1S, test_data_output_1S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 1st Floor Winter  
plt.figure(figsize=(12,5))
plt.title('1st-Floor Winter, [R_2:%s] ' %w_1W_t, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_1W, y_hat_1W_t,'bo',markersize=5,label='Prediction')
plt.plot(Time_1W, test_data_output_1W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 2nd Floor Summer  
plt.figure(figsize=(12,5))
plt.title('2nd-Floor Summer, [R_2:%s] ' %w_2S_t, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2S, y_hat_2S_t,'bo',markersize=5,label='Prediction')
plt.plot(Time_2S, test_data_output_2S,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 2nd Floor Winter  
plt.figure(figsize=(12,5))
plt.title('2nd-Floor Winter, [R_2:%s] ' %w_2W_t, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2W, y_hat_2W_t,'bo',markersize=5,label='Prediction')
plt.plot(Time_2W, test_data_output_2W,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 3rd Floor  
plt.figure(figsize=(12,5))
plt.title('3rd-Floor, [R_2:%s] ' %w_3_t, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_3, y_hat_3_t,'bo',markersize=5,label='Prediction')
plt.plot(Time_3, test_data_output_3,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)

#4th Floor vs 4th Floor  
plt.figure(figsize=(12,5))
plt.title('4th-Floor, [R_2:%s] ' %w_4, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_4, y_hat_4,'bo',markersize=5,label='Prediction')
plt.plot(Time_4, test_data_output_4,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)


