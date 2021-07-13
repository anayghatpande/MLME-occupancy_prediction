# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:36:35 2021

@author: Felix
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import requests as req
import sklearn
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

#Data cleaning stuff.. this time importing the hohle excel data and saves all the sheets in a dictionary
#the cleaning process happens for all the data in the dictionary.

df_building = pd.read_excel('occupancy_data.xlsx',sheet_name = None)

#convert the dict to a list 
list_1 = list(df_building.values()) 

#iterating through all dataframes in the list to drop Unnamed column
for n in range(len(list_1)):
    list_1[n] = list_1[n].drop(columns = ['Unnamed: 0'])

#Iterate through rows of all dataframes in the list to interpolate the NaN's
for k in list_1:
    for i in range(1, len(k.columns)):
        for j in range(1, len(k)-1):
            if np.isnan(k.iloc[j,i]):
                k.iloc[j,i]=(k.iloc[j-1,i]+k.iloc[j+1,i])/2
                
#getting a list with all the heders              
list_keys = list(df_building.keys())

#realizing that dict's would have been better to work with ...so convert it back to a dictionary
df_building_clean = dict(zip(list_keys,list_1))

#now, getting a dict with all the heders of all the dataframes
dct_ind = {}
for l in df_building.keys():
    dct_ind[l] = []

#getting the indeces of the night-data 
ind_time = {} 
dct_times = {}
for v in df_building_clean.keys():
    dct_times[v] = df_building_clean.get(v).iloc[:,0]
    ind_time[v] = (np.ones(len(dct_times[v])))

# removing night-data from all dataframes 
for df_1, df_2 in ind_time.items():
    n_1 = ind_time[df_1]
    n_2 = dct_times[df_1]
    n_3 = dct_ind[df_1]
    for k_1 in range(len(n_1)):
        n_1[k_1] = n_2[k_1].hour
        if n_1[k_1] < 6 or n_1[k_1] > 18:
            n_3.append(k_1)
        
for df_3, df_4 in df_building_clean.items():
    df_building_clean[df_3] = df_4.drop(index = dct_ind[df_3])

#Having finally all the cleaned dataframes in one dictionary which gives me acces to all the possible training data in a easy and fast way


Floor1S_heads = list(df_building_clean.get('Floor1S').columns)
Floor1W_heads = list(df_building_clean.get('Floor1W').columns)
Floor2S_heads = list(df_building_clean.get('Floor2S').columns)
Floor2W_heads = list(df_building_clean.get('Floor2W').columns)
Floor3_heads = list(df_building_clean.get('Floor3').columns)
Floor4_heads = list(df_building_clean.get('Floor4').columns)


#Plotting all the clean data for first overview
plot=df_building_clean.get('Floor1S').plot(x=Floor1S_heads[0],y=Floor1S_heads[1:len(Floor1S_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor1S')
plot=df_building_clean.get('Floor1W').plot(x=Floor1W_heads[0],y=Floor1W_heads[1:len(Floor1W_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor1W')
plot=df_building_clean.get('Floor2S').plot(x=Floor2S_heads[0],y=Floor2S_heads[1:len(Floor2S_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor2S')
plot=df_building_clean.get('Floor2W').plot(x=Floor1S_heads[0],y=Floor2W_heads[1:len(Floor2W_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor2W')
plot=df_building_clean.get('Floor3').plot(x=Floor3_heads[0],y=Floor3_heads[1:len(Floor3_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor3')
plot=df_building_clean.get('Floor4').plot(x=Floor4_heads[0],y=Floor4_heads[1:len(Floor4_heads)],subplots=False, marker='.', figsize=(50,30), grid=True, title='Floor4')    


#create Excel file
with pd.ExcelWriter('Clean_set_2.xlsx') as writer:

    df_building_clean.get('Floor1S').to_excel(writer, sheet_name='Floor1S')
    df_building_clean.get('Floor1W').to_excel(writer, sheet_name='Floor1W')
    df_building_clean.get('Floor2S').to_excel(writer, sheet_name='Floor2S')
    df_building_clean.get('Floor2W').to_excel(writer, sheet_name='Floor2W')
    df_building_clean.get('Floor3').to_excel(writer, sheet_name='Floor3')
    df_building_clean.get('Floor4').to_excel(writer, sheet_name='Floor4')

    