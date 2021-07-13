import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
##import requests as req
import sklearn
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tkinter import *
while(True):
    val = input("Enter Floor name: ")
    print(val)
    df_building = pd.read_excel('Clean_Set.xlsx',sheet_name = None)

    df_building_clean = {}
    for k1 in df_building.keys():
        df_building_clean[k1] = df_building.get(k1).drop('Unnamed: 0',axis=1)
        
        
    Floor_heads = list(df_building_clean.get(val).columns)
    dfg = df_building_clean[val]
    

    df_building_clean[val] = df_building_clean.get(val).drop(columns = ['Time'])
    df_building_clean.get(val).insert(1, 'PIR_ALL',df_building_clean.get(val)[Floor_heads[1:27]].sum(axis=1), True)
    df_building_clean.get(val).insert(1, 'CO2_ALL',df_building_clean.get(val).reindex(columns=Floor_heads[28:53]).sum(axis=1), True)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled1 = min_max_scaler.fit_transform(df_building_clean.get(val)[['CO2_ALL']])
    df_building_clean.get(val).insert(1, 'CO2_ALL_scaled',x_scaled1, True)
    
    dataset = df_building_clean[val]
    

    #AP_Total
    x1 = dataset["AP_Total"].values.reshape(-1, 1)
    y1 = dataset["Groundtruth"].values.reshape(-1, 1)

    regressor = LinearRegression()  
    regressor.fit(x1, y1) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y1_pred = regressor.predict(x1)

    df1 = pd.DataFrame({'Actual': y1.flatten(), 'Predicted': y1_pred.flatten()})
    print('mean squared error',mean_squared_error(y1, y1_pred))
    print("AP_Total ",r2_score(y1, y1_pred))
    

    plt.scatter(x1, y1,  color='blue')
    plt.plot(x1, y1_pred, color='red', linewidth=2)
    plt.title('AP_Total '+str(val)+' R2 value '+str(r2_score(y1, y1_pred)))
    plt.xlabel("AP_Total")
    plt.ylabel('Groundtruth')
    plt.show()

    #"Inst_kW_Load_Plug"
    x1_1 = dataset["Inst_kW_Load_Plug"].values.reshape(-1, 1)
    y1_1 = dataset["Groundtruth"].values.reshape(-1, 1)

    
    regressor = LinearRegression()  
    regressor.fit(x1_1, y1_1) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y1_1_pred = regressor.predict(x1_1)

    df1_1 = pd.DataFrame({'Actual': y1_1.flatten(), 'Predicted': y1_1_pred.flatten()})
    print('mean squared error',mean_squared_error(y1_1, y1_1_pred))
    print("Inst_kW_Load_Plug ",r2_score(y1_1, y1_1_pred))
    

    plt.scatter(x1_1, y1_1,  color='blue')
    plt.plot(x1_1, y1_1_pred, color='red', linewidth=2)
    plt.title('Inst_kW_Load_Plug '+str(val)+' R2 value '+str(r2_score(y1_1, y1_1_pred)))
    plt.xlabel("Inst_kW_Load_Plug")
    plt.ylabel('Groundtruth')
    plt.show()


    #PIR_ALL
    x1_2 = dataset["PIR_ALL"].values.reshape(-1, 1)
    y1_2 = dataset["Groundtruth"].values.reshape(-1, 1)

    
    regressor = LinearRegression()  
    regressor.fit(x1_2, y1_2) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y1_2_pred = regressor.predict(x1_2)

    df1_2 = pd.DataFrame({'Actual': y1_2.flatten(), 'Predicted': y1_2_pred.flatten()})
    print('mean squared error',mean_squared_error(y1_2, y1_2_pred))
    print("PIR_ALL ",r2_score(y1_2, y1_2_pred))
    

    plt.scatter(x1_2, y1_2,  color='blue')
    plt.plot(x1_2, y1_2_pred, color='red', linewidth=2)
    plt.title('PIR_ALL '+str(val)+' R2 value '+str(r2_score(y1_2, y1_2_pred)))
    plt.xlabel("PIR_ALL")
    plt.ylabel('Groundtruth')
    plt.show()

    #CO2_ALL
    x2 = dataset["CO2_ALL"].values.reshape(-1, 1)
    y2 = dataset["Groundtruth"].values.reshape(-1, 1)

    #x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(x2, y2) #training the algorithm
    

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y2_pred = regressor.predict(x2)

    df2 = pd.DataFrame({'Actual': y2.flatten(), 'Predicted': y2_pred.flatten()})
    

    
    print("CO2_ALL",r2_score(y2, y2_pred))
    
    plt.scatter(x2, y2,  color='blue')
    plt.plot(x2, y2_pred, color='red', linewidth=2)
    plt.title("CO2_ALL "+str(val) +' R2 value '+str(r2_score(y2, y2_pred)))
    plt.xlabel("CO2_ALL")
    plt.ylabel('Groundtruth')
    plt.show()

    #"AP_Total","Inst_kW_Load_Plug"
    x3 = dataset[["AP_Total","Inst_kW_Load_Plug"]].values
    y3 = dataset["Groundtruth"].values

    

    regressor = LinearRegression()  
    regressor.fit(x3, y3) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y3_pred = regressor.predict(x3)

    df3 = pd.DataFrame({'Actual': y3.flatten(), 'Predicted': y3_pred.flatten()})

    
    print("AP_Total+Inst_kW_Load_Plug ",r2_score(y3, y3_pred))


    #"AP_Total","Inst_kW_Load_Plug","PIR_ALL"
    x4 = dataset[["AP_Total","Inst_kW_Load_Plug","PIR_ALL"]].values
    y4 = dataset["Groundtruth"].values

    

    regressor = LinearRegression()  
    regressor.fit(x4, y4) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y4_pred = regressor.predict(x4)

    df4 = pd.DataFrame({'Actual': y4.flatten(), 'Predicted': y4_pred.flatten()})

    
    print("AP_Total+Inst_kW_Load_Plug+PIR_ALL ",r2_score(y4, y4_pred))
    


    x5 = dataset[["AP_Total","Inst_kW_Load_Plug","PIR_ALL","CO2_ALL"]].values
    y5 =dataset["Groundtruth"].values

    
    regressor = LinearRegression()  
    regressor.fit(x5, y5) #training the algorithm

    #To retrieve the intercept:
    print(regressor.intercept_)

    #For retrieving the slope:
    print(regressor.coef_)

    y5_pred = regressor.predict(x5)

    df5 = pd.DataFrame({'Actual': y5.flatten(), 'Predicted': y1_pred.flatten()})
    print('mean squared error',mean_squared_error(y5, y5_pred))
    print("AP_Total+Inst_kW_Load_Plug+PIR_ALL+CO2_ALL ",r2_score(y5, y5_pred))





