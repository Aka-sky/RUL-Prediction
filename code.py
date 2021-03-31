import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd

def getTime(h, m, s, ms):
    return h * 3600 + m * 60 + s + ms * (10 ** -6)

def load_train_data(condition, bearing):
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
    for i in range(1,2803):
        num = str(i)
        zero_filled = num.zfill(5)
        with open(os.path.join('ieee-phm-2012-data-challenge-dataset-master','Learning_set','Bearing1_1','acc_' + zero_filled + '.csv')) as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if(first):
                    start = getTime(int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                    t.append(0)
                    v.append(float(row[5]))
                    first = 0
                else:
                    t.append(round(getTime(int(row[0]), int(row[1]), int(row[2]), float(row[3])) - start, 6))
                    v.append(float(row[5]))
                    if(abs(float(row[5])) > 20):
                        life_time = round(getTime(int(row[0]), int(row[1]), int(row[2]), float(row[3])) - start, 6)
                        found = 1
                        break
        if(found):
            break
    print('Loaded Dataset')
    # Shrinking the data from t and v to time and v_acc
    shrinkBy = 100
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
    print('Shrunk by',shrinkBy)
    return time, v_acc, life_time

def preprocess_dataset(time, v_acc, life_time, window_size=100):
    # rolling rms -------------------
    rolling_rms = pd.Series(v_acc).pow(2).rolling(window_size).apply(lambda x: np.sqrt(x.mean()))
    rolling_rms = rolling_rms.dropna()   
    print("Calculated Rolling RMS")
    
    # rolling kurtosis -------------------
    rolling_kurt = pd.Series(v_acc)
    rolling_kurt = rolling_kurt.rolling(window_size).kurt()
    rolling_kurt = rolling_kurt.dropna()
    print("Calculated Rolling Kurtosis")
    
    # rolling time ------------------
    rolling_time = pd.Series(time)
    rolling_time = rolling_time.rolling(window_size).mean()
    rolling_time = rolling_time.dropna()
    print("Calculated Rolling Time")
    
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
    print("Calculated Weibull RMS")
    
    # Weibull Hazard for Kurtosis -------------------
    weibull_hazardKurt = []
    for i in rolling_kurt:
        if(i > 0):
            weibull_hazardKurt.append(round(etaByGammaKurt * ((i / gammaKurt) ** (etaKurt - 1)), 6))
        else:
            weibull_hazardKurt.append(0)
    print("Calculated Weibull Kurtosis")
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

#------------------------------------------Train Bearings---------------------------------------

time, v_acc, life_time = load_train_data(condition=1,bearing=1)

time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time, window_size = 100) 

X_train, Y_train = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)

# plt.plot(time,v_acc)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()

# ieee-phm-2012-data-challenge-dataset-master\Learning_set\Bearing1_1
