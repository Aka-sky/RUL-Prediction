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

######################################### Condition 1 ###########################################
#---------------------------------------Loading Bearings---------------------------------------
print('Condition 1: \n')

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

sc_1 = StandardScaler()
X_train1_1 = sc_1.fit_transform(X_train1_1)
X_train1_2 = sc_1.fit_transform(X_train1_2)
X_test1_3 = sc_1.fit_transform(X_test1_3)
#X_test1_4 = sc.transform(X_test1_4)
X_test1_5 = sc_1.fit_transform(X_test1_5)
X_test1_6 = sc_1.fit_transform(X_test1_6)
X_test1_7 = sc_1.fit_transform(X_test1_7)
print('\nApplied Feature Scaling\n')

#-----------------------------------Standard Library ANN Model-------------------------------------

regressor_1 = Sequential()
regressor_1.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_1.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_1.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_1.fit(X_train1_1, Y_train1_1, batch_size=1000, epochs=100)
print('C1B1: Model Trained')
regressor_1.fit(X_train1_2, Y_train1_2, batch_size=1000, epochs=100)
print('C1B2: Model Trained')

Y_pred1_3 = regressor_1.predict(X_test1_3)
mean_percent_error = np.mean((abs(Y_pred1_3.flatten() - Y_test1_3) / Y_test1_3) * 100)
print('Test B13 Mean Error: ',mean_percent_error,'%')

#Y_pred1_4 = regressor.predict(X_test1_4)
#mean_percent_error = np.mean((abs(Y_pred1_4.flatten() - Y_test1_4) / Y_test1_4) * 100)
#print('Test B14 Mean Error: ',mean_percent_error,'%')

Y_pred1_5 = regressor_1.predict(X_test1_5)
mean_percent_error = np.mean((abs(Y_pred1_5.flatten() - Y_test1_5) / Y_test1_5) * 100)
print('Test B15 Mean Error: ',mean_percent_error,'%')

Y_pred1_6 = regressor_1.predict(X_test1_6)
mean_percent_error = np.mean((abs(Y_pred1_6.flatten() - Y_test1_6) / Y_test1_6) * 100)
print('Test B16 Mean Error: ',mean_percent_error,'%')

Y_pred1_7 = regressor_1.predict(X_test1_7)
mean_percent_error = np.mean((abs(Y_pred1_7.flatten() - Y_test1_7) / Y_test1_7) * 100)
print('Test B17 Mean Error: ',mean_percent_error,'%')


######################################### Condition 2 ###########################################
#---------------------------------------Loading Bearings---------------------------------------
print('Condition 2: \n')

time, v_acc, life_time = load_data(learning=True, condition=2, bearing=1, filelength=911)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train2_1, Y_train2_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=2, bearing=2, filelength=797)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train2_2, Y_train2_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B2: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=2, bearing=3, filelength=1955)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test2_3, Y_test2_3 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B3: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=2, bearing=4, filelength=751)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test2_4, Y_test2_4 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B4: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=2, bearing=5, filelength=2311)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test2_5, Y_test2_5 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B5: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=2, bearing=6, filelength=701)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test2_6, Y_test2_6 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B6: Test Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=2, bearing=7, filelength=230)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test2_7, Y_test2_7 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C2B7: Test Dataset Loaded')

#-----------------------------------------Feature Scaling-----------------------------------------

sc_2 = StandardScaler()
X_train2_1 = sc_2.fit_transform(X_train2_1)
X_train2_2 = sc_2.fit_transform(X_train2_2)
X_test2_3 = sc_2.fit_transform(X_test2_3)
X_test2_4 = sc_2.fit_transform(X_test2_4)
X_test2_5 = sc_2.fit_transform(X_test2_5)
X_test2_6 = sc_2.fit_transform(X_test2_6)
X_test2_7 = sc_2.fit_transform(X_test2_7)
print('\nApplied Feature Scaling\n')

#-----------------------------------Standard Library ANN Model-------------------------------------

regressor_2 = Sequential()
regressor_2.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_2.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_2.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_2.fit(X_train2_1, Y_train2_1, batch_size=1000, epochs=1000)
print('C2B1: Model Trained')
regressor_2.fit(X_train2_2, Y_train2_2, batch_size=1000, epochs=1000)
print('C2B2: Model Trained')

Y_pred2_3 = regressor_2.predict(X_test2_3)
mean_percent_error = np.mean((abs(Y_pred2_3.flatten() - Y_test2_3) / Y_test2_3) * 100)
print('Test B13 Mean Error: ',mean_percent_error,'%')

Y_pred2_4 = regressor_2.predict(X_test2_4)
mean_percent_error = np.mean((abs(Y_pred2_4.flatten() - Y_test2_4) / Y_test2_4) * 100)
print('Test B14 Mean Error: ',mean_percent_error,'%')

Y_pred2_5 = regressor_2.predict(X_test2_5)
mean_percent_error = np.mean((abs(Y_pred2_5.flatten() - Y_test2_5) / Y_test2_5) * 100)
print('Test B15 Mean Error: ',mean_percent_error,'%')

Y_pred2_6 = regressor_2.predict(X_test2_6)
mean_percent_error = np.mean((abs(Y_pred2_6.flatten() - Y_test2_6) / Y_test2_6) * 100)
print('Test B16 Mean Error: ',mean_percent_error,'%')

Y_pred2_7 = regressor_2.predict(X_test2_7)
mean_percent_error = np.mean((abs(Y_pred2_7.flatten() - Y_test2_7) / Y_test2_7) * 100)
print('Test B17 Mean Error: ',mean_percent_error,'%')


######################################### Condition 3 ###########################################
#---------------------------------------Loading Bearings---------------------------------------
print('Condition 3: \n')

time, v_acc, life_time = load_data(learning=True, condition=3, bearing=1, filelength=515)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train3_1, Y_train3_1 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C3B1: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=True, condition=3, bearing=2, filelength=1637)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_train3_2, Y_train3_2 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C3B2: Train Dataset Loaded')

time, v_acc, life_time = load_data(learning=False, condition=3, bearing=3, filelength=434)
time, weibull_RMS, weibull_Kurt = preprocess_dataset(time, v_acc, life_time) 
X_test3_3, Y_test3_3 = split_x_y(life_time, time, weibull_RMS, weibull_Kurt)
print('C3B3: Test Dataset Loaded')

#-----------------------------------------Feature Scaling-----------------------------------------

sc_3 = StandardScaler()
X_train3_1 = sc_3.fit_transform(X_train3_1)
X_train3_2 = sc_3.fit_transform(X_train3_2)
X_test3_3 = sc_3.fit_transform(X_test3_3)
print('\nApplied Feature Scaling\n')

#-----------------------------------Standard Library ANN Model-------------------------------------

regressor_3 = Sequential()
regressor_3.add(Dense(input_dim=6, output_dim=2, activation='sigmoid', init='uniform'))
regressor_3.add(Dense(output_dim=1, activation='relu', init='uniform'))
regressor_3.compile(optimizer='adam', loss='mean_absolute_percentage_error')
regressor_3.fit(X_train3_1, Y_train3_1, batch_size=1000, epochs=1000)
print('C3B1: Model Trained')
regressor_3.fit(X_train3_2, Y_train3_2, batch_size=1000, epochs=1000)
print('C3B2: Model Trained')

Y_pred3_3 = regressor_3.predict(X_test3_3)
mean_percent_error = np.mean((abs(Y_pred3_3.flatten() - Y_test3_3) / Y_test3_3) * 100)
print('Test B13 Mean Error: ',mean_percent_error,'%')

#---------------------------------------Our ANN Model---------------------------------------------
"""
EPSILON = 10 ** -8

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)     # no. of layers including hidden & o/p layer

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def relu(Z):
    return np.maximum(Z, 0)
    
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def softmax(z):
    z = np.exp(z-np.max(z, axis=0))
    return z / z.sum(axis=0)

def relu_backward(dAL, Z):
    Z[Z < 0] = 0
    Z[Z >= 0] = 1
    dZ = np.multiply(dAL, Z)
    return dZ

def sigmoid_backward(dAL, Z):
    dZ = np.multiply(dAL, sigmoid(Z), 1-sigmoid(Z))
    return dZ

def softmax_backward(dAL, AL):
    dZ = np.multiply(dAL, AL, 1-AL)
    return dZ

def compute_cost(AL, Y):
    # cost = -np.sum(Y * np.log(AL + EPSILON))
    cost = np.mean((AL-Y)**2) / 2
    return cost

def activation_forward(A, W, b, activation_type):
    Z = np.dot(W, A) + b 
    linear_cache = (A, W, b)

    if activation_type == 'sigmoid':
        A = sigmoid(Z)
    elif activation_type == 'softmax':
        A = softmax(Z)
    elif activation_type == 'relu':
        A = relu(Z)
    
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2    # no. of layers
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "sigmoid")
        caches.append(cache)
    AL, cache = activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "relu")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    current_cache = caches[L-1]
    dZ = AL - Y
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, current_cache[0])
    for l in reversed(range(L-1)):
        current_cache = caches[l]   
        dZ = linear_backward(grads["dA"+str(l+1)], current_cache[1])
        dA_prev_temp, dW_temp, db_temp = sigmoid_backward(dZ, current_cache[0]) 
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate*grads["db"+str(l+1)]
    return parameters
    
def plot_cost(costs):
    # Plot learning curve (with costs)
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Training Plot")
    plt.show()

class ANN:
    layer_dims = []
    learning_rate=0.0075
    batch_size=1000
    num_iterations=1000
    parameters = {}
    
    def __init__(self, layer_dims, learning_rate=0.0075, batch_size=1000, num_iterations=3000):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iterations  = num_iterations
    
    def fit(self, X, Y, print_cost=True):
        costs = []
        Y = Y.reshape(-1,1)
    
        self.parameters = initialize_parameters(self.layer_dims)
    
        batches = [(X[:, i:i + self.batch_size], Y[:, i:i + self.batch_size])
                       for i in range(0, X.shape[1], self.batch_size)]
    
        # Gradient descent
        for i in range(self.num_iterations):
        
            for x_, y_ in batches:
    
                AL, caches = forward_propagation(x_, self.parameters)
    
                cost = compute_cost(AL, y_)
    
                grads = backward_propagation(AL, y_, caches)
    
                self.parameters = update_parameters(self.parameters, grads, self.learning_rate)
    
            if print_cost and i % 1 == 0:
                print ("Iteration : {} Cost : {}".format(i, cost), sep='\t')
                costs.append(cost)
    
        return costs
    
    
    def predict(self, X):
        AL, _ = forward_propagation(X, self.parameters)
        predictions = np.argmax(AL, axis=0)
        return predictions

ann = ANN(layer_dims = [6, 2, 1], learning_rate=0.0075, batch_size=1000, num_iterations=1000)
costs = ann.fit(X_train1_1, Y_train1_1)
Y_pred1_3 = ann.predict(X_test1_3)
mean_percent_error = np.mean((abs(Y_pred1_3.flatten() - Y_test1_3) / Y_test1_3) * 100)
print('Test B13 Mean Error: ',mean_percent_error,'%')
"""
