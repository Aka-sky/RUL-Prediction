import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#-------------------------------------------Data Preprocessing----------------------------------------------

def getTime(h, m, s, ms):
    return h * 3600 + m * 60 + s + ms * (10 ** -6)

def load_data(learning, condition, bearing, filelength, shrinkBy = 100):
    subfolder = 'Full_Test_Set'
    if(learning):
        subfolder = 'Learning_set'
    # These are used for input calculations ------
    time = []
    v_acc = []
    life_time = 0
    # These are temporary
    t = []
    v = []
    found = 0
    first = 1
    start = 0.0
    for i in range(1,filelength):
        num = str(i)
        zero_filled = num.zfill(5)
        with open(os.path.join('ieee-phm-2012-data-challenge-dataset-master',subfolder,'Bearing'+str(condition)+'_'+str(bearing),'acc_' + zero_filled + '.csv')) as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if(first):
                    start = getTime(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    t.append(0)
                    v.append(float(row[5]))
                    first = 0
                else:
                    t.append(round(getTime(float(row[0]), float(row[1]), float(row[2]), float(row[3])) - start, 6))
                    v.append(float(row[5]))
                    if(abs(float(row[5])) > 20):
                        life_time = round(getTime(float(row[0]), float(row[1]), float(row[2]), float(row[3])) - start, 6)
                        found = 1
                        break
        if(found):
            break
    #print('Loaded Dataset')
    # Shrinking the data from t and v to time and v_acc
    itr = int(len(t) / shrinkBy)
    remainingEle = len(t) % shrinkBy 
    # print("Remaining elements: ",remainingEle)
    for i in range(itr):
        startIndex = i * shrinkBy
        endIndex = startIndex + shrinkBy
        time.append(np.mean(t[startIndex:endIndex]))
        maxEle = 0
        for j in range(startIndex, endIndex):
            if(abs(v[i]) > abs(maxEle)):
                maxEle = v[i]
        v_acc.append(maxEle)
    # For remaining elements if any
    if (remainingEle > 0):
        time.append(np.mean(t[itr * shrinkBy : ]))
        maxEle = 0
        for j in range(itr * shrinkBy, len(v)):
            if(abs(v[j]) > abs(maxEle)):
                maxEle = v[j]
        v_acc.append(maxEle)
    #print('Shrunk by',shrinkBy)
    if(life_time == 0):
        life_time = time[-1]
    return time, v_acc, life_time

def preprocess_dataset(time, v_acc, life_time, window_size=100):
    # rolling rms -------------------
    rolling_rms = pd.Series(v_acc).pow(2).rolling(window_size).apply(lambda x: np.sqrt(x.mean()))
    rolling_rms = rolling_rms.dropna()   
    #print("Calculated Rolling RMS")
    
    # rolling kurtosis -------------------
    rolling_kurt = pd.Series(v_acc)
    rolling_kurt = rolling_kurt.rolling(window_size).kurt()
    rolling_kurt = rolling_kurt.dropna()
    #print("Calculated Rolling Kurtosis")
    
    # rolling time ------------------
    rolling_time = pd.Series(time)
    rolling_time = rolling_time.rolling(window_size).mean()
    rolling_time = rolling_time.dropna()
    #print("Calculated Rolling Time")
    
    # Weibull constants -------------------
    # for RMS
    etaRMS = 1.2017
    gammaRMS = 0.4077
    etaByGammaRMS = etaRMS / gammaRMS
    # for Kurtosis
    etaKurt = 1.2970
    gammaKurt = 0.4360
    etaByGammaKurt = etaKurt / gammaKurt
    
    # Weibull Hazard for RMS ------------------
    weibull_hazardRMS = []
    for i in rolling_rms:
        if(i > 0):
            weibull_hazardRMS.append(round(etaByGammaRMS * ((i / gammaRMS) ** (etaRMS - 1)), 6))
        else: 
            weibull_hazardRMS.append(0)
    #print("Calculated Weibull RMS")
    
    # Weibull Hazard for Kurtosis -------------------
    weibull_hazardKurt = []
    for i in rolling_kurt:
        if(i > 0):
            weibull_hazardKurt.append(round(etaByGammaKurt * ((i / gammaKurt) ** (etaKurt - 1)), 6))
        else:
            weibull_hazardKurt.append(0)
    #print("Calculated Weibull Kurtosis")
    return rolling_time.tolist(), weibull_hazardRMS, weibull_hazardKurt

def split_x_y(life_time, time, weibull_RMS, weibull_Kurt):
    x_train = np.empty((1,6), float)
    y_train = np.array([round(i/life_time,6) for i in time])
    x = np.array([time, weibull_RMS, weibull_Kurt]).transpose()
    for i in range(1,x.shape[0]):
        x_train = np.append(x_train, np.reshape(np.hstack((x[i-1],x[i])), (-1, 6)), axis=0)
        
    x_train = x_train[1:]
    y_train = y_train[1:]
    return x_train, y_train

#------------------------------------Loading Bearings in condition 1---------------------------------------

time, v_acc, life_time = load_data(learning=True, condition=1, bearing=1, filelength=2803)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train1_1, Y_train1_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=1, bearing=2, filelength=871)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train1_2, Y_train1_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B2: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=1, bearing=3, filelength=2375)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test1_3, Y_test1_3 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B3: Test Dataset Loaded')

#time, v_acc, life_time = load_data(learning=False, condition=1, bearing=4, filelength=1428)
#time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
#X_test1_4, Y_test1_4 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
#print('C1B4: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=1, bearing=5, filelength=2463)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test1_5, Y_test1_5 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B5: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=1, bearing=6, filelength=2448)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test1_6, Y_test1_6 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B6: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=1, bearing=7, filelength=2259)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test1_7, Y_test1_7 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B7: Test Dataset Loaded')

#-----------------------------------------Feature Scaling-----------------------------------------

sc = StandardScaler()
X_train1_1 = sc.fit_transform(X_train1_1)
X_train1_2 = sc.fit_transform(X_train1_2)
X_test1_3 = sc.fit_transform(X_test1_3)
#X_test1_4 = sc.transform(X_test1_4)
X_test1_5 = sc.fit_transform(X_test1_5)
X_test1_6 = sc.fit_transform(X_test1_6)
X_test1_7 = sc.fit_transform(X_test1_7)

#-----------------------------------Standard Library ANN Model-------------------------------------
regressor = Sequential()
regressor.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor.fit(X_train1_1, Y_train1_1, batch_size=1000, epochs=1000)
print('C1B1: Model Trained')
regressor.fit(X_train1_2, Y_train1_2, batch_size=1000, epochs=1000)
print('C1B2: Model Trained')

Y_pred1_3 = regressor.predict(X_test1_3)
mean_percent_error = np.mean((abs(Y_pred1_3.flatten() - Y_test1_3) / Y_test1_3) * 100)
print('Test B13 Mean Error: ',mean_percent_error,'%')

#Y_pred1_4 = regressor.predict(X_test1_4)
#mean_percent_error = np.mean((abs(Y_pred1_4.flatten() - Y_test1_4) / Y_test1_4) * 100)
#print('Test B14 Mean Error: ',mean_percent_error,'%')

Y_pred1_5 = regressor.predict(X_test1_5)
mean_percent_error = np.mean((abs(Y_pred1_5.flatten() - Y_test1_5) / Y_test1_5) * 100)
print('Test B15 Mean Error: ',mean_percent_error,'%')

Y_pred1_6 = regressor.predict(X_test1_6)
mean_percent_error = np.mean((abs(Y_pred1_6.flatten() - Y_test1_6) / Y_test1_6) * 100)
print('Test B16 Mean Error: ',mean_percent_error,'%')

Y_pred1_7 = regressor.predict(X_test1_7)
mean_percent_error = np.mean((abs(Y_pred1_7.flatten() - Y_test1_7) / Y_test1_7) * 100)
print('Test B17 Mean Error: ',mean_percent_error,'%')
