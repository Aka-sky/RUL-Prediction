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

def preprocess_dataset(time, v_acc, window_size=100):
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

#---------------------------------------Neural Net----------------------------------------

class NeuralNet():
    #A two layer neural network

    def __init__(self, layers=[6, 2, 1], learning_rate=0.001, iterations=100, batch_size = 1000):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
        self.batch_size = batch_size

    def init_weights(self):
        #np.random.seed(1)  # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )

    def relu(self, Z):
        return np.maximum(0, Z)

    def dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def eta(self, x):
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat)
        yhat_inv = self.eta(yhat_inv)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self, x_, i):
        
        Z1 = x_.dot(self.params['W1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.relu(Z2)
        #loss = self.entropy_loss(self.y, yhat)
#        print("\nx: ",x_[:5,:])
#        print("\nZ1: ",Z1[:5,:])
#        print("\nA1: ",A1[:5,:])
#        print("\nZ2: ",Z2[:2,:])
#        print("\nyhat: ",yhat[:2,:])

        self.params['Z1'+str(i)] = Z1
        self.params['Z2'+str(i)] = Z2
        self.params['A1'+str(i)] = A1

        return yhat.reshape(yhat.shape[0])

    def back_propagation(self, x_, y_, yhat, i):
        # y_inv = 1 - y_
        # yhat_inv = 1 - yhat
        # np.divide(y_inv, self.eta(yhat_inv)) - np.divide(y_, self.eta(yhat))
#        print("\ni ",i)
#        print("\nx ",x_[:5,:])
#        print("\nyhat: ",yhat[:5])
        dl_wrt_yhat = yhat - y_
        dl_wrt_z2 = dl_wrt_yhat * self.dRelu(self.params['Z2'+str(i)].flatten())
        dl_wrt_z2 = dl_wrt_z2.reshape(dl_wrt_z2.shape[0],1)
#        print("dl_wrt_yhat_shape: ",dl_wrt_yhat.shape)
#        print("dl_wrt_z2_shape: ",dl_wrt_z2.shape)
        
#        print("\ndl_wrt_yhat: ",dl_wrt_yhat[:2])
#        print("\nz2: ",self.params['Z2'+str(i)])
#        print("\ndl_wrt_z2: ",dl_wrt_z2[:2])

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'+str(i)].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

#        print("\ndl_wrt_A1: ",dl_wrt_A1[:2])
#        print("\ndl_wrt_w2: ",dl_wrt_w2[:2])
#        print("\ndl_wrt_b2: ",dl_wrt_b2[:2])

        dl_wrt_z1 = dl_wrt_A1 * self.params['A1'+str(i)] * (np.ones(self.params['A1'+str(i)].shape) - self.params['A1'+str(i)])
        dl_wrt_w1 = x_.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)
        
#        print("\ndl_wrt_z1: ",dl_wrt_z1[:2])
#        print("\ndl_wrt_w1: ",dl_wrt_w1[:2])
#        print("\ndl_wrt_b1: ",dl_wrt_b1[:2])

        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2


    def fit(self, X, y):
        self.X = X
        self.y = y
        # initialize weights and bias
        self.init_weights()  
        
        batches = [(X[i:i + self.batch_size,:], y[i:i + self.batch_size]) for i in range(0, X.shape[0], self.batch_size)]
        #print("batches_shape: ",len(batches))

        for i in range(self.iterations):
            for i in range(len(batches)):
                x_, y_ = batches[i]
                #print("x_shape: ",x_.shape)
                #print("y_shape: ",y_.shape)
                yhat = self.forward_propagation(x_, i)
                #print("yhat_shape: ",yhat.shape)
                self.back_propagation(x_, y_, yhat, i)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

# nn1 = NeuralNet(layers=[6, 2, 1], learning_rate=0.01, iterations=100)
# nn1.fit(X_train1_1, Y_train1_1)
# print('C1B1: Model Trained')
#nn1.fit(X_train1_2, Y_train1_2)
#print('C1B2: Model Trained')
#
#Y_pred1_3 = nn1.predict(X_test1_3)
#mean_percent_error = np.mean((abs(Y_pred1_3.flatten() - Y_test1_3) / Y_test1_3) * 100)
#predicted_RUL = (1-Y_pred1_3[-1])[0]*life_time1_3
#print('\nB13 Predicted RUL: ',predicted_RUL,'s')
#print('B13 Actual RUL: ',test_RUL1_3,'s')
#print('Test B13 Mean Error: ',mean_percent_error,'%')
#---------------------------------------Test RULs-------------------------------------------------------

test_RUL1_3 = 5730
test_RUL1_5 = 1610
test_RUL1_6 = 1460
test_RUL1_7 = 7570
test_RUL2_3 = 7530
test_RUL2_4 = 1390
test_RUL2_5 = 3090
test_RUL2_6 = 1290
test_RUL2_7 = 580
test_RUL3_3 = 820

#---------------------------------------Loading and Preprocessing---------------------------------------
        
print('\nCondition 1: \n')

time, v_acc, life_time = load_data(learning=True, condition=1, bearing=1, filelength=2803)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train1_1, Y_train1_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=1, bearing=2, filelength=871)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train1_2, Y_train1_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C1B2: Train Dataset Loaded')

time, v_acc, life_time1_3 = load_data(learning=False, condition=1, bearing=3, filelength=2375)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time1_3 += test_RUL1_3
X_test1_3, Y_test1_3 = split_x_y(life_time1_3, time, weibull_RMS, weibull_Kurt)
print('C1B3: Test Dataset Loaded')

#time, v_acc, life_time = load_data(learning=False, condition=1, bearing=4, filelength=1428)
#time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
#X_test1_4, Y_test1_4 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
#print('C1B4: Test Dataset Loaded')

time, v_acc, life_time1_5 = load_data(learning=False, condition=1, bearing=5, filelength=2463)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time1_5 += test_RUL1_5
X_test1_5, Y_test1_5 = split_x_y(life_time1_5, time, weibull_RMS, weibull_Kurt)
print('C1B5: Test Dataset Loaded')

time, v_acc, life_time1_6 = load_data(learning=False, condition=1, bearing=6, filelength=2448)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time1_6 += test_RUL1_6
X_test1_6, Y_test1_6 = split_x_y(life_time1_6, time, weibull_RMS, weibull_Kurt)
print('C1B6: Test Dataset Loaded')

time, v_acc, life_time1_7 = load_data(learning=False, condition=1, bearing=7, filelength=2259)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time1_7 += test_RUL1_7
X_test1_7, Y_test1_7 = split_x_y(life_time1_7, time, weibull_RMS, weibull_Kurt)
print('C1B7: Test Dataset Loaded')


print('\nCondition 2: \n')

time, v_acc, life_time = load_data(learning=True, condition=2, bearing=1, filelength=911)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train2_1, Y_train2_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=2, bearing=2, filelength=797)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train2_2, Y_train2_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B2: Train Dataset Loaded')

time, v_acc, life_time2_3 = load_data(learning=False, condition=2, bearing=3, filelength=1955)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time2_3 += test_RUL2_3
X_test2_3, Y_test2_3 = split_x_y(life_time2_3, time, weibull_RMS, weibull_Kurt)
print('C2B3: Test Dataset Loaded')

time, v_acc, life_time2_4 = load_data(learning=False, condition=2, bearing=4, filelength=751)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time2_4 += test_RUL2_4
X_test2_4, Y_test2_4 = split_x_y(life_time2_4, time, weibull_RMS, weibull_Kurt)
print('C2B4: Test Dataset Loaded')

time, v_acc, life_time2_5 = load_data(learning=False, condition=2, bearing=5, filelength=2311)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time2_5 += test_RUL2_5
X_test2_5, Y_test2_5 = split_x_y(life_time2_5, time, weibull_RMS, weibull_Kurt)
print('C2B5: Test Dataset Loaded')

time, v_acc, life_time2_6 = load_data(learning=False, condition=2, bearing=6, filelength=701)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time2_6 += test_RUL2_6
X_test2_6, Y_test2_6 = split_x_y(life_time2_6, time, weibull_RMS, weibull_Kurt)
print('C2B6: Test Dataset Loaded')

time, v_acc, life_time2_7 = load_data(learning=False, condition=2, bearing=7, filelength=230)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time2_7 += test_RUL2_7
X_test2_7, Y_test2_7 = split_x_y(life_time2_7, time, weibull_RMS, weibull_Kurt)
print('C2B7: Test Dataset Loaded')


print('\nCondition 3: \n')

time, v_acc, life_time = load_data(learning=True, condition=3, bearing=1, filelength=515)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train3_1, Y_train3_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C3B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=3, bearing=2, filelength=1637)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
X_train3_2, Y_train3_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C3B2: Train Dataset Loaded')

time, v_acc, life_time3_3 = load_data(learning=False, condition=3, bearing=3, filelength=434)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc) 
life_time3_3 += test_RUL3_3
X_test3_3, Y_test3_3 = split_x_y(life_time3_3, time, weibull_RMS, weibull_Kurt)
print('C3B3: Test Dataset Loaded')

#-----------------------------------------Feature Scaling-----------------------------------------

sc_1 = StandardScaler()
X_train1_1 = sc_1.fit_transform(X_train1_1)
X_train1_2 = sc_1.fit_transform(X_train1_2)
X_test1_3 = sc_1.fit_transform(X_test1_3)
#X_test1_4 = sc.transform(X_test1_4)
X_test1_5 = sc_1.fit_transform(X_test1_5)
X_test1_6 = sc_1.fit_transform(X_test1_6)
X_test1_7 = sc_1.fit_transform(X_test1_7)

sc_2 = StandardScaler()
X_train2_1 = sc_2.fit_transform(X_train2_1)
X_train2_2 = sc_2.fit_transform(X_train2_2)
X_test2_3 = sc_2.fit_transform(X_test2_3)
X_test2_4 = sc_2.fit_transform(X_test2_4)
X_test2_5 = sc_2.fit_transform(X_test2_5)
X_test2_6 = sc_2.fit_transform(X_test2_6)
X_test2_7 = sc_2.fit_transform(X_test2_7)

sc_3 = StandardScaler()
X_train3_1 = sc_3.fit_transform(X_train3_1)
X_train3_2 = sc_3.fit_transform(X_train3_2)
X_test3_3 = sc_3.fit_transform(X_test3_3)
print('\nApplied Feature Scaling\n')

#-----------------------------------Training Models-------------------------------------

regressor_1 = Sequential()
regressor_1.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_1.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_1.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_1.fit(X_train1_1, Y_train1_1, batch_size=1000, epochs=40)
print('C1B1: Model Trained')
regressor_1.fit(X_train1_2, Y_train1_2, batch_size=1000, epochs=40)
print('C1B2: Model Trained')

#nn1 = NeuralNet(layers=[6, 2, 1], learning_rate=0.01, iterations=100)
#nn1.fit(X_train1_1, Y_train1_1)
#print('C1B1: Model Trained')
#nn1.fit(X_train1_2, Y_train1_2)
#print('C1B2: Model Trained')


regressor_2 = Sequential()
regressor_2.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_2.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_2.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_2.fit(X_train2_1, Y_train2_1, batch_size=1000, epochs=40)
print('C2B1: Model Trained')
regressor_2.fit(X_train2_2, Y_train2_2, batch_size=1000, epochs=40)
print('C2B2: Model Trained')


regressor_3 = Sequential()
regressor_3.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_3.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_3.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_3.fit(X_train3_1, Y_train3_1, batch_size=1000, epochs=40)
print('C3B1: Model Trained')
regressor_3.fit(X_train3_2, Y_train3_2, batch_size=1000, epochs=40)
print('C3B2: Model Trained')

#-----------------------------------------Results---------------------------------------------

print('\nCondition 1:')

Y_pred1_3 = regressor_1.predict(X_test1_3)
mean_percent_error = np.mean((abs(Y_pred1_3.flatten() - Y_test1_3) / Y_test1_3) * 100)
predicted_RUL = (1-Y_pred1_3[-1])[0]*life_time1_3
print('\nB13 Predicted RUL: ',predicted_RUL,'s')
print('B13 Actual RUL: ',test_RUL1_3,'s')
print('Test B13 Mean Error: ',mean_percent_error,'%')

#Y_pred1_4 = regressor.predict(X_test1_4)
#mean_percent_error = np.mean((abs(Y_pred1_4.flatten() - Y_test1_4) / Y_test1_4) * 100)
#print('Test B14 Mean Error: ',mean_percent_error,'%')

Y_pred1_5 = regressor_1.predict(X_test1_5)
mean_percent_error = np.mean((abs(Y_pred1_5.flatten() - Y_test1_5) / Y_test1_5) * 100)
predicted_RUL = (1-Y_pred1_5[-1])[0]*life_time1_5
print('\nB15 Predicted RUL: ',predicted_RUL,'s')
print('B15 Actual RUL: ',test_RUL1_5,'s')
print('Test B15 Mean Error: ',mean_percent_error,'%')

Y_pred1_6 = regressor_1.predict(X_test1_6)
mean_percent_error = np.mean((abs(Y_pred1_6.flatten() - Y_test1_6) / Y_test1_6) * 100)
predicted_RUL = (1-Y_pred1_6[-1])[0]*life_time1_6
print('\nB16 Predicted RUL: ',predicted_RUL,'s')
print('B16 Actual RUL: ',test_RUL1_6,'s')
print('Test B16 Mean Error: ',mean_percent_error,'%')

Y_pred1_7 = regressor_1.predict(X_test1_7)
mean_percent_error = np.mean((abs(Y_pred1_7.flatten() - Y_test1_7) / Y_test1_7) * 100)
predicted_RUL = (1-Y_pred1_7[-1])[0]*life_time1_7
print('\nB17 Predicted RUL: ',predicted_RUL,'s')
print('B17 Actual RUL: ',test_RUL1_7,'s')
print('Test B17 Mean Error: ',mean_percent_error,'%')

print('\nCondition 2:')

Y_pred2_3 = regressor_2.predict(X_test2_3)
mean_percent_error = np.mean((abs(Y_pred2_3.flatten() - Y_test2_3) / Y_test2_3) * 100)
predicted_RUL = (1-Y_pred2_3[-1])[0]*life_time2_3
print('\nB23 Predicted RUL: ',predicted_RUL,'s')
print('B23 Actual RUL: ',test_RUL2_3,'s')
print('Test B23 Mean Error: ',mean_percent_error,'%')

Y_pred2_4 = regressor_2.predict(X_test2_4)
mean_percent_error = np.mean((abs(Y_pred2_4.flatten() - Y_test2_4) / Y_test2_4) * 100)
predicted_RUL = (1-Y_pred2_4[-1])[0]*life_time2_4
print('\nB24 Predicted RUL: ',predicted_RUL,'s')
print('B24 Actual RUL: ',test_RUL2_4,'s')
print('Test B24 Mean Error: ',mean_percent_error,'%')

Y_pred2_5 = regressor_2.predict(X_test2_5)
mean_percent_error = np.mean((abs(Y_pred2_5.flatten() - Y_test2_5) / Y_test2_5) * 100)
predicted_RUL = (1-Y_pred2_5[-1])[0]*life_time2_5
print('\nB25 Predicted RUL: ',predicted_RUL,'s')
print('B25 Actual RUL: ',test_RUL2_5,'s')
print('Test B25 Mean Error: ',mean_percent_error,'%')

Y_pred2_6 = regressor_2.predict(X_test2_6)
mean_percent_error = np.mean((abs(Y_pred2_6.flatten() - Y_test2_6) / Y_test2_6) * 100)
predicted_RUL = (1-Y_pred2_6[-1])[0]*life_time2_6
print('\nB26 Predicted RUL: ',predicted_RUL,'s')
print('B26 Actual RUL: ',test_RUL2_6,'s')
print('Test B26 Mean Error: ',mean_percent_error,'%')

Y_pred2_7 = regressor_2.predict(X_test2_7)
mean_percent_error = np.mean((abs(Y_pred2_7.flatten() - Y_test2_7) / Y_test2_7) * 100)
predicted_RUL = (1-Y_pred2_7[-1])[0]*life_time2_7
print('\nB27 Predicted RUL: ',predicted_RUL,'s')
print('B27 Actual RUL: ',test_RUL2_7,'s')
print('Test B27 Mean Error: ',mean_percent_error,'%')

print('\nCondition 3:')

Y_pred3_3 = regressor_3.predict(X_test3_3)
mean_percent_error = np.mean((abs(Y_pred3_3.flatten() - Y_test3_3) / Y_test3_3) * 100)
predicted_RUL = (1-Y_pred3_3[-1])[0]*life_time3_3
print('\nB33 Predicted RUL: ',predicted_RUL,'s')
print('B33 Actual RUL: ',test_RUL3_3,'s')
print('Test B33 Mean Error: ',mean_percent_error,'%\n')

#--------------------------------------Plots-------------------------------------------

time = [i for i in range(Y_test1_3.shape[0])]
plt.plot(time,Y_test1_3)
plt.plot(time,Y_pred1_3)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing1_3')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test1_5.shape[0])]
plt.plot(time,Y_test1_5)
plt.plot(time,Y_pred1_5)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing1_5')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test1_6.shape[0])]
plt.plot(time,Y_test1_6)
plt.plot(time,Y_pred1_6)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing1_6')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test1_7.shape[0])]
plt.plot(time,Y_test1_7)
plt.plot(time,Y_pred1_7)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing1_7')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test2_3.shape[0])]
plt.plot(time,Y_test2_3)
plt.plot(time,Y_pred2_3)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing2_3')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test2_4.shape[0])]
plt.plot(time,Y_test2_4)
plt.plot(time,Y_pred2_4)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing2_4')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test2_5.shape[0])]
plt.plot(time,Y_test2_5)
plt.plot(time,Y_pred2_5)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing2_5')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test2_6.shape[0])]
plt.plot(time,Y_test2_6)
plt.plot(time,Y_pred2_6)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing2_6')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test2_7.shape[0])]
plt.plot(time,Y_test2_7)
plt.plot(time,Y_pred2_7)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing2_7')
plt.legend(['Actual','Predicted'])
plt.show()

time = [i for i in range(Y_test3_3.shape[0])]
plt.plot(time,Y_test3_3)
plt.plot(time,Y_pred3_3)
plt.xlabel('Time in sec')
plt.ylabel('Completed Life percentage')
plt.title('For Bearing3_3')
plt.legend(['Actual','Predicted'])
plt.show()

