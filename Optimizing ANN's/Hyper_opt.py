# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:27:26 2021

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


#loading data
df_building = pd.read_excel('Clean_Set.xlsx',sheet_name = None)

#dropping unused column
df_building_clean = {}
for k1 in df_building.keys():
    df_building_clean[k1] = df_building.get(k1).drop('Unnamed: 0',axis=1)
    
#getting a list of all heads for easy indexing later     
Floor1S_heads = list(df_building_clean.get('Floor1S').columns)
Floor1W_heads = list(df_building_clean.get('Floor1W').columns)
Floor2S_heads = list(df_building_clean.get('Floor2S').columns)
Floor2W_heads = list(df_building_clean.get('Floor2W').columns)
Floor3_heads = list(df_building_clean.get('Floor3').columns)
Floor4_heads = list(df_building_clean.get('Floor4').columns)

#using the head-lists to sum up the right  columns
df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[28:53]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(0, 'PIR_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(1, 'CO2_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(0, 'PIR_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(1, 'CO2_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)





#get input sets
input_data1 = df_building_clean.get('Floor1S')[['AP_Total']]
output_data1 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input1 = df_building_clean.get('Floor2W')[['AP_Total']]
test_data_output1 = df_building_clean.get('Floor2W')[['Groundtruth']]

input_data2 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total']]
output_data2 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input2 = df_building_clean.get('Floor2W')[['Inst_kW_Load_Plug','AP_Total']]
test_data_output2 = df_building_clean.get('Floor2W')[['Groundtruth']]

input_data3 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
output_data3 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input3 = df_building_clean.get('Floor2W')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
test_data_output3 = df_building_clean.get('Floor2W')[['Groundtruth']]

input_data5 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL','PIR_ALL']]
output_data5 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input5 = df_building_clean.get('Floor2W')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL','PIR_ALL']]
test_data_output5 = df_building_clean.get('Floor2W')[['Groundtruth']]


line = [1,2,3,5]
pipeit = 0

#choose the correct input for experiments
for pipe in line:
    pipeit = 1 + pipeit
    if pipe == 1:
        input_data = input_data1
        output_data = output_data1
        test_data_input = test_data_input1
        test_data_output = test_data_output1
    elif pipe == 2:
        input_data = input_data2
        output_data = output_data2
        test_data_input = test_data_input2
        test_data_output = test_data_output2
    elif pipe == 3:
        input_data = input_data3
        output_data = output_data3
        test_data_input = test_data_input3
        test_data_output = test_data_output3
    # elif pipe == 4:
    #     input_data = input_data4
    #     output_data = output_data4
    #     test_data_input = test_data_input4
    #     test_data_output = test_data_output4
    elif pipe == 5:
        input_data = input_data5
        output_data = output_data5
        test_data_input = test_data_input5
        test_data_output = test_data_output5
        
        
    #validation split on input data
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=99, test_size=0.2)
    
    #get input and test data transformations    
    input_scaler = preprocessing.StandardScaler().fit(x_train)
    output_scaler = preprocessing.StandardScaler().fit(y_train)
    
    
    
    
    
    
    
    #creating keras model including as function
    def get_keras_model(h_layers,h_neurones, activation, m, n):
        
        inputs = keras.Input(shape=x_train.shape[1], name='input')
    
        layer_output = [inputs]
       
        for n_l in range(h_layers):
            
            layer_output.append(keras.layers.Dense(h_neurones,activation=activation,name='hidden_{}'.format(n_l))(layer_output[-1]))
            
            layer_output.append(keras.layers.Dropout(n)(layer_output[n_l-1]))
            
        layer_output.append(keras.layers.Dense(y_train.shape[1], name='output')(layer_output[-1]))
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=layer_output[-1]
            )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=m),
            loss=tf.keras.losses.MSE,

            )        
        return model
    
    
    #creating nested loops for grid search. 
    h_layers = [1,2]#,3,4]
    h_neurones = [10,70]#,100,200]
    activation = ['relu']#,'tanh','sigmoid']

    ep = [50]#,30,70,100, alternativ a high number was combined with early stopping
    ba = [200]#,100,70,30]
    rate = [0.01]#,0.05
    drop = [0]#,0.2,0.4
    it=1
    
    #creating dataframe to store results in
    my_grid_df = pd.DataFrame(columns=['Hidden_Layers','Neurones','Activation','Epoches','Batch','learning_rate','drop_rate','loss','val_loss','Fit[R_2]_train','Fit[R_2]_cross'])
    
    for i in h_layers:
        
        for h in h_neurones:
            
            for a in activation:
                
                for j in ep:
                                    
                    for l in ba:
                       
                        for m in rate:
                            
                            for n in drop:
                                
                                #Print the Progress in '%'
                                print(100 * (it/(len(h_layers)*len(h_neurones)*len(activation)*len(drop)*len(ep)*len(ba)*len(rate))))
                                it = it+1
                                
                                #Call model and specify inputs using loop's
                                gt_model = get_keras_model(i,h,a,m,n)
                                
                                history = gt_model.fit(
                                    input_scaler.transform(x_train),
                                    output_scaler.transform(y_train),
                                    batch_size = l,
                                    epochs = j,
                                    validation_data=(input_scaler.transform(x_test), output_scaler.transform(y_test)),
                                    verbose=0
                                    )

                                #get loss and val_loss to add them to dataframe
                                acc = history.history['loss']
                                val_loss = history.history['val_loss']
                                
                                #get prediction performance with validation set
                                y_hat_scaled = gt_model.predict(input_scaler.transform(x_test))
                                y_hat = output_scaler.inverse_transform(y_hat_scaled)
                                w_1 = sklearn.metrics.r2_score(y_hat, y_test)
                                
                                #get prediction performance with different dataset
                                y_hat_scaled2 = gt_model.predict(input_scaler.transform(test_data_input))
                                y_hat2 = output_scaler.inverse_transform(y_hat_scaled2)
                                w_2 = sklearn.metrics.r2_score(y_hat2, test_data_output)
                                
                                #create new dataframe to safe results of the iteration
                                df2 = pd.DataFrame(
                                    [[i,h,a,j,l,m,n,acc[-1],val_loss[-1],w_1,w_2]],
                                    columns=['Hidden_Layers','Neurones','Activation','Epoches','Batch','learning_rate','drop_rate','loss','val_loss','Fit[R_2]_train','Fit[R_2]_cross']
                                    )
                                
                                #append new dataframe to my_grid_df
                                my_grid_df = my_grid_df.append(df2,ignore_index=True)
                        
                
    
    #save my_grid_df to excel
    my_grid_df.to_excel('grid_search_%s.xlsx'%pipeit)
    input_data.to_excel('grid_data_%s.xlsx'%pipeit)




# h_layers = [2,3,4,5]
# h_neurones = [85,95,100,105,115]
# activation = ['relu']
# j = 100
# l = 250
# it=1
# my_grid_df_plot = pd.DataFrame(columns=['Hidden_Layers','Neurones','Fit[R_2]-1S','Fit[R_2]-2W'])

# for i in h_layers:
    
    
#     for h in h_neurones:
        
#         for a in activation:
            
           
                
                
             
#                 print( (it/(len(h_layers)*len(h_neurones))))
                
            
#                 gt_model = get_keras_model(i,h,a,0.01,0)
#                 history = gt_model.fit(
#                     input_scaler.transform(x_train),
#                     output_scaler.transform(y_train),
#                     batch_size = l,
#                     epochs = j,
#                     validation_data=(input_scaler.transform(x_test), output_scaler.transform(y_test)),
#                     verbose=0
#                     )
                
#                 y_hat_scaled = gt_model.predict(input_scaler.transform(x_test))
#                 y_hat = output_scaler.inverse_transform(y_hat_scaled)
#                 w_1 = sklearn.metrics.r2_score(y_hat, y_test)
                
#                 y_hat_scaled2 = gt_model.predict(input_scaler.transform(test_data_input))
#                 y_hat2 = output_scaler.inverse_transform(y_hat_scaled2)
#                 w_2 = sklearn.metrics.r2_score(y_hat2, test_data_output)
                
#                 df2 = pd.DataFrame([[i,h,w_1,w_2]], columns=['Hidden_Layers','Neurones','Fit[R_2]-1S','Fit[R_2]-2W'])
#                 my_grid_df_plot = my_grid_df_plot.append(df2,ignore_index=True)
#                 it = it+1
                
# X=my_grid_df_plot.iloc[:,0]
# Y=my_grid_df_plot.iloc[:,1]
# Z=my_grid_df_plot.iloc[:,2]
                    
                
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection='3d')
# ax.set_title("Grid_search-1S: epochs: %s, batch: %s, Activation : %s" %(j, l, activation))

# ax.plot_trisurf(X,Y, Z,cmap=plt.cm.Spectral)

# ax.set_xlabel("Hidden_Layers")

# ax.set_ylabel("Neurones")

# ax.set_zlabel("Fit [R_2]-2S")

# ax.grid(True)

            
# Z2=my_grid_df_plot.iloc[:,3]


# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection='3d')
# ax.set_title("Grid_search-2W: epochs: %s, batch: %s, Activation : %s" %(j, l, activation))

# ax.plot_trisurf(X,Y, Z2,cmap=plt.cm.Spectral)

# ax.set_xlabel("Hidden_Layers")

# ax.set_ylabel("Neurones")

# ax.set_zlabel("Fit [R_2]-2W")

# ax.grid(True)