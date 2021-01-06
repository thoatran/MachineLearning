from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
from time import time 

print('LVQ classifier, Digits dataset')

# np.random.seed(7)

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
x_min = np.min(X_test)
x_max = np.max(X_test)
# print('Test set:',X_test.shape[0])

print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])
print('Label:',np.unique(y_test))

# locate the best matching prototype 
def get_best_matching_prototype (x,prototype_set):
    dis_list = list()
    for xyp in prototype_set:
        dis = np.linalg.norm(x-xyp[:-1]) 
        dis_list.append((xyp,dis))
    dis_list.sort(key=lambda tup: tup[1])
    return dis_list[0][0]

# create a random prototype vector 
def create_random_prototype(data_train,idx):
    n_records = len(data_train)
    n_features = len(data_train[0])
    xy_prototype = [data_train[np.random.randint(n_records)][i] for i in range(n_features-1)]
    xy_prototype += [idx%len(np.unique(y_test))]
    return xy_prototype

# training a set of prototype vectors 
def train_prototypes(data_train, n_prototypes, lrate_init, n_epochs):
    prototype_set = [create_random_prototype(data_train,i) for i in range(n_prototypes)]
    err_vec = []
    for epoch in range(n_epochs):
        lrate = lrate_init * (1.0 - epoch/float(n_epochs))
        sum_err = 0.0
        for xy_train in data_train:
            bmu = get_best_matching_prototype(xy_train[:-1], prototype_set) # bmu is a view of the prototype_set
            err = xy_train[:-1] - bmu[:-1]
            sum_err += np.linalg.norm(1/16*err)**2 
            # if bmu[-1] == xy_train[-1]: 
            #     bmu[:-1] += lrate*err 
            # else:
            #     bmu[:-1] -= lrate*err 
            if bmu[-1] == xy_train[-1]: 
                bmu[:-1] += lrate*err 
            else:
                # bmu[:-1] -= lrate*err 
                for i in range(len(bmu[:-1])):
                    tmp = bmu[i] - lrate*err[i]
                    if tmp < x_min:
                        bmu[i] = x_min
                    elif tmp > x_max:
                        bmu[i] = x_max
                    else:
                        bmu[i] = tmp
        print('>epoch=%d, lrate=%.6f, err=%.3f'%(epoch,lrate,sum_err))
        err_vec += [sum_err]
    return (prototype_set,err_vec)

    
# main(): 
n_run = 1
lrate_init = 0.3
n_epochs = 100
n_prototypes = 100 # 10 clusters
accuracy_vec = np.zeros(n_run) 
ttrain_vec = np.zeros(n_run)
ttest_vec = np.zeros(n_run)  
for i_run in range(n_run):
    # w = evaluate(lrate_init,n_epochs,n_prototypes)

    # lvq training
    ts_train = time()
    # Xy_prototype,err_vec = np.array(train_prototypes(Xy_train, n_prototypes, lrate_init, n_epochs),dtype=object)
    tmp1,tmp2 = train_prototypes(Xy_train, n_prototypes, lrate_init, n_epochs)
    Xy_prototype = np.array(tmp1)
    err_vec = np.array(tmp2)
    ttrain_vec[i_run] = time() - ts_train
    # print(Xy_prototype[0])
    # print(np.unique(Xy_prototype[...,-1]))

    # predict outputs 
    predict_list = list()
    ts_test = time()
    for x_test in X_test:
        # lvq algo
        predict_list.append(get_best_matching_prototype(x_test,Xy_prototype)[-1])
    ttest_vec[i_run] = time()-ts_test
    y_predict = np.array(predict_list)

    # evaluate the prediction
    e = 0
    for i in range(len(y_predict)): 
        if y_predict[i] != y_test[i]:
            e += 1

    accuracy_vec[i_run] = 100*(1-e/len(y_test))
    print('irun =', i_run, ', accuracy =', np.round(accuracy_vec[i_run],2), 'err =',np.round(err_vec[-1],2))

print('accuracy:',np.round(accuracy_vec[:10],2))
# print('train time:',np.round(ttrain_vec,5))
# print('test time:',np.round(ttest_vec,5))
print('average accuracy: %.2f%%'%(np.average(accuracy_vec)))
print('average train time: %.5fs'%(np.average(ttrain_vec)))
print('average test time: %.5fs'%(np.average(ttest_vec)))
accuracy_avg = np.average(accuracy_vec)
rtime_avg = np.average(ttest_vec)

# confution matrix 
cm = confusion_matrix(y_test,y_predict) 
cm_df = pd.DataFrame(cm,
                        index=['0', '1', '2','3','4','5','6','7','8','9'],
                        columns=['0', '1', '2','3','4','5','6','7','8','9'])

plt.figure(figsize=(7, 6))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title("LVQ Classifier, Digits dataset \nAvg. accuracy: {:.2f}%,  Avg. running time: {:.3f} (s)".format(accuracy_avg, rtime_avg))
# plt.title('1NN Classifier, Iris dataset \nAccuracy: %.2f%, Running time: %.2f (s)'%(accuracy_avg, rtime_avg))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# error figure
plt.figure(figsize=(7, 6))
plt.plot(err_vec)
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid()
plt.show()