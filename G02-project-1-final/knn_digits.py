# m5232108 Hoang Tuan Linh

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('1NN classifier, Digits dataset')

# Xy_train_pd = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
Xy_train_pd = pd.read_csv('/Users/linhht20/Google Drive/ITC05F Machine Learning/optdigits.tra',header=None)
Xy_train = np.array(Xy_train_pd)
X_train = np.delete(Xy_train,-1,1)
y_train = np.copy(Xy_train[...,-1])
# print('Training set:',X_train.shape[0])

# Xy_test_pd = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
Xy_test_pd = pd.read_csv('/Users/linhht20/Google Drive/ITC05F Machine Learning/optdigits.tes',header=None)
Xy_test = np.array(Xy_test_pd)
X_test = np.delete(Xy_test,-1,1)
y_test = np.copy(Xy_test[...,-1])
# print('Test set:',X_test.shape[0])

Xy_train = np.concatenate((X_train,y_train.reshape(1,-1).T), axis=1)

print('Label:', np.unique(y_test))
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])

# locate the best matching prototype 
def get_best_matching_prototype (x,prototype_set):
    dis_list = list()
    for xyp in prototype_set:
        dis = np.linalg.norm(x-xyp[:-1]) 
        dis_list.append((xyp,dis))
    dis_list.sort(key=lambda tup: tup[1])
    return dis_list[0][0]

# main(): 
n_test = 10
print('Number of runs:',n_test)
accuracy_vec = list()
rtime_vec = list()
for i in range(n_test):
    print('irun=',i)
    # # 1NN (k=1)
    # model = neighbors.KNeighborsClassifier(n_neighbors=7, p=2) # p=2->norm2
    # model.fit(X_train, y_train)

    # # predict outputs 
    # t_start = time()
    # y_pred = model.predict(X_test)
    # rtime_vec.append(time()-t_start)

    # predict outputs 
    predict_list = list()
    ts_test = time()
    for x_test in X_test:
        # 1NN 
        predict_list.append(get_best_matching_prototype(x_test,Xy_train)[-1])
    rtime_vec.append(time()-ts_test)
    y_pred = np.array(predict_list)

    # evaluate accuracy
    accuracy_vec.append(100*accuracy_score(y_test, y_pred))

# print('Accuracy:',np.round(acc_vec,2))
accuracy_avg = np.average(accuracy_vec)
print('Average accuracy:  %.2f %%' %accuracy_avg)
accuracy_var = np.var(accuracy_vec)
print('Accuracy variance:  %.2f %%' % accuracy_var)
rtime_avg = np.average(rtime_vec)
print('Average running time: %.5f (s)'%rtime_avg)

# plot accurracy vs irun 
# error figure
plt.figure(figsize=(7, 6))
plt.plot(accuracy_vec)
plt.xlabel('Index of run')
plt.ylabel('Accuracy (%)')
plt.title('1NN classifier, Digits dataset')
plt.grid()
plt.show()

# confution matrix 
cm = confusion_matrix(y_test,y_pred) 
cm_df = pd.DataFrame(cm,
                        index=['0', '1', '2','3','4','5','6','7','8','9'],
                        columns=['0', '1', '2','3','4','5','6','7','8','9'])

plt.figure(figsize=(7, 6))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title("1NN Classifier, Digits dataset \nAvg. accuracy: {:.2f}%, Avg. running time: {:.3f} (s)".format(accuracy_avg, rtime_avg))
# plt.title('1NN Classifier, Iris dataset \nAccuracy: %.2f%, Running time: %.2f (s)'%(accuracy_avg, rtime_avg))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()