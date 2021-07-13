# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:11:49 2021

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
Time_2S = df_building_clean.get('Floor1S')[['Time']]

# df_building_clean['Floor1S'] = df_building_clean.get('Floor1S').drop(columns= Floor1S_heads[1:27])
df_building_clean['Floor1S'] = df_building_clean.get('Floor1S').drop(columns = ['AP1','AP2','AP3'])
df_building_clean['Floor1S'] = df_building_clean.get('Floor1S').drop(columns = ['Inst_kW_Load_Light','Inst_kW_Load_Elec'])
# df_building_clean['Floor1S'] = df_building_clean.get('Floor1S').iloc[:,14:30]

df_building_clean['Floor1W'] = df_building_clean.get('Floor1W').drop(columns = ['AP1','AP2','AP3'])
df_building_clean['Floor1W'] = df_building_clean.get('Floor1W').drop(columns = ['Inst_kW_Load_Light','Inst_kW_Load_Elec'])

# df_building_clean['Floor2S'] = df_building_clean.get('Floor2S').drop(columns= Floor2S_heads[1:11])
df_building_clean['Floor2S'] = df_building_clean.get('Floor2S').drop(columns = ['AP1','AP2','AP3','AP4','AP5'])
df_building_clean['Floor2S'] = df_building_clean.get('Floor2S').drop(columns = ['Inst_kW_Load_Light','Inst_kW_Load_Elec'])
df_building_clean['Floor2S'] = df_building_clean.get('Floor2S').drop(columns = ['Time'])

df_building_clean['Floor2W'] = df_building_clean.get('Floor2W').drop(columns = ['AP1','AP2','AP3','AP4','AP5'])
df_building_clean['Floor2W'] = df_building_clean.get('Floor2W').drop(columns = ['Inst_kW_Load_Light','Inst_kW_Load_Elec'])
df_building_clean['Floor2W'] = df_building_clean.get('Floor2W').drop(columns = ['Time'])



df_building_clean.get('Floor1S').insert(1, 'PIR_ALL',df_building_clean.get('Floor1S')[Floor1S_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL',df_building_clean.get('Floor1S').loc[:,Floor1S_heads[28:53]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(0, 'PIR_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2S').insert(1, 'CO2_ALL',df_building_clean.get('Floor2S').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(0, 'PIR_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[1:11]].sum(axis=1), True)
df_building_clean.get('Floor2W').insert(1, 'CO2_ALL',df_building_clean.get('Floor2W').loc[:,Floor2S_heads[12:24]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'PIR_ALL',df_building_clean.get('Floor1W')[Floor1W_heads[1:27]].sum(axis=1), True)
df_building_clean.get('Floor1W').insert(1, 'CO2_ALL',df_building_clean.get('Floor1W').loc[:,Floor1W_heads[28:53]].sum(axis=1), True)


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled1 = min_max_scaler.fit_transform(df_building_clean.get('Floor1S')[['CO2_ALL']])
df_building_clean.get('Floor1S').insert(1, 'CO2_ALL_scaled',x_scaled1, True)
x_scaled1 = min_max_scaler.fit_transform(df_building_clean.get('Floor1W')[['CO2_ALL']])
df_building_clean.get('Floor1W').insert(1, 'CO2_ALL_scaled',x_scaled1, True)
x_scaled = min_max_scaler.fit_transform(df_building_clean.get('Floor2S')[['CO2_ALL']])
df_building_clean.get('Floor2S').insert(1, 'CO2_ALL_scaled',x_scaled, True)
x_scaled = min_max_scaler.fit_transform(df_building_clean.get('Floor2W')[['CO2_ALL']])
df_building_clean.get('Floor2W').insert(1, 'CO2_ALL_scaled',x_scaled, True)

#get Input and test data
input_data1 = df_building_clean.get('Floor1S')[['AP_Total']]
output_data1 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input1 = df_building_clean.get('Floor1W')[['AP_Total']]
test_data_output1 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data2 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total']]
output_data2 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input2 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total']]
test_data_output2 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data3 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
output_data3 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input3 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL']]
test_data_output3 = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL','CO2_ppm_101','CO2_ppm_102','CO2_ppm_103','CO2_ppm_104','CO2_ppm_105','CO2_ppm_106','CO2_ppm_107','CO2_ppm_108','CO2_ppm_109','CO2_ppm_110','CO2_ppm_111','CO2_ppm_112','CO2_ppm_113','CO2_ppm_114','CO2_ppm_115','CO2_ppm_116','CO2_ppm_117','CO2_ppm_118','CO2_ppm_119','CO2_ppm_120','CO2_ppm_121','CO2_ppm_122','CO2_ppm_123','CO2_ppm_124','CO2_ppm_125','CO2_ppm_126']]
output_data = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','PIR_ALL','CO2_ppm_101','CO2_ppm_102','CO2_ppm_103','CO2_ppm_104','CO2_ppm_105','CO2_ppm_106','CO2_ppm_107','CO2_ppm_108','CO2_ppm_109','CO2_ppm_110','CO2_ppm_111','CO2_ppm_112','CO2_ppm_113','CO2_ppm_114','CO2_ppm_115','CO2_ppm_116','CO2_ppm_117','CO2_ppm_118','CO2_ppm_119','CO2_ppm_120','CO2_ppm_121','CO2_ppm_122','CO2_ppm_123','CO2_ppm_124','CO2_ppm_125','CO2_ppm_126']]
test_data_output = df_building_clean.get('Floor1W')[['Groundtruth']]

input_data5 = df_building_clean.get('Floor1S')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL_scaled']]
output_data5 = df_building_clean.get('Floor1S')[['Groundtruth']]
test_data_input5 = df_building_clean.get('Floor1W')[['Inst_kW_Load_Plug','AP_Total','CO2_ALL_scaled']]
test_data_output5 = df_building_clean.get('Floor1W')[['Groundtruth']]

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=99, test_size=0.2)
    
#get input and test data transformations    
input_scaler = preprocessing.StandardScaler().fit(x_train)
output_scaler = preprocessing.StandardScaler().fit(y_train)
    





reconstructed_model = keras.models.load_model("Model_1stF_allInputs")








y_hat_scaled = reconstructed_model.predict(input_scaler.transform(input_data))
y_hat = output_scaler.inverse_transform(y_hat_scaled)
w_1 = sklearn.metrics.r2_score(y_hat, output_data)






plt.figure(figsize=(12,5))
plt.title('Model 1st-FloorS vs. Data 1st-FloorS, [R_2:%s] ' %w_1, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2S, y_hat,'b',markersize=5,label='Prediction')
plt.plot(Time_2S, output_data,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)




y_hat_scaled2 = reconstructed_model.predict(input_scaler.transform(test_data_input))
y_hat2 = output_scaler.inverse_transform(y_hat_scaled2)
w_2 = sklearn.metrics.r2_score(y_hat2, test_data_output)





plt.figure(figsize=(12,5))
plt.title('Model 1st-FloorS vs. Data 1st-FloorW, [R_2:%s] ' %w_2, fontsize=15, color= 'black', y= 1.1)
plt.plot(Time_2W, y_hat2,'bo',markersize=3,label='Prediction')
plt.plot(Time_2W, test_data_output,'ro',markersize=3,label='Groundtruth')
plt.legend(loc='upper center')
plt.xlabel('Time')
plt.ylabel('Occupancy-count')
plt.grid(True)