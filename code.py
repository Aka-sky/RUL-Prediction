import matplotlib.pyplot as plt
import os
import csv

x = []
y = []
j = 1

for i in range(1,2803,10):
    num = str(i)
    zero_filled = num.zfill(5)
    with open(os.path.join('ieee-phm-2012-data-challenge-dataset-master','Learning_set','Bearing1_1','acc_' + zero_filled + '.csv')) as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(j)
            y.append(row[5])
            j = j + 1

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

# ieee-phm-2012-data-challenge-dataset-master\Learning_set\Bearing1_1