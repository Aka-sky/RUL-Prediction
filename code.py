import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd

def getTime(h, m, s, ms):
    return h * 3600 + m * 60 + s + ms * (10 ** -6)

time = []
v_acc = []

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
                time.append(0)
                v_acc.append(float(row[5]))
                first = 0
            else:
                time.append(round(getTime(int(row[0]), int(row[1]), int(row[2]), float(row[3])) - start, 6))
                v_acc.append(float(row[5]))
                if(abs(float(row[5])) > 20):
                    # print(zero_filled + "cmkdmcksk")
                    found = 1
                    break
    if(found):
        break

window_size = 100

# rolling rms -------------------
rolling_rms = pd.Series(v_acc).pow(2).rolling(window_size).apply(lambda x: np.sqrt(x.mean()))
rolling_rms = rolling_rms.dropna()   
print("Rolling RMS: " , len(rolling_rms))

# rolling kurtosis -------------------
rolling_kurt = pd.Series(v_acc)
rolling_kurt = rolling_kurt.rolling(window_size).kurt()
rolling_kurt = rolling_kurt.dropna()
print("Rolling Kurtosis: " , len(rolling_kurt))

# rolling time ------------------
rolling_time = pd.Series(time)
rolling_time = rolling_time.rolling(window_size).mean()
rolling_time = rolling_time.dropna()
print("Rolling Time: " , len(rolling_time))

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
print("Weibull RMS: " , len(weibull_hazardRMS))

# Weibull Hazard for Kurtosis -------------------
weibull_hazardKurt = []
for i in rolling_kurt:
    if(i > 0):
        weibull_hazardKurt.append(round(etaByGammaKurt * ((i / gammaKurt) ** (etaKurt - 1)), 6))
    else:
        weibull_hazardKurt.append(0)
print("Weibull Kurtosis: " , len(weibull_hazardKurt))

# plt.plot(time,v_acc)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()

# ieee-phm-2012-data-challenge-dataset-master\Learning_set\Bearing1_1
