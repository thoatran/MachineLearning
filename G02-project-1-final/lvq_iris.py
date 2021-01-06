from __future__ import print_function
import numpy as np 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

print('LVQ classifier, Iris dataset')
# np.random.seed(7)

# load iris data 
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target 
# print('Labels:', np.unique(iris_y))

def split_data(test_sz): 
    # split training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=test_sz)

    # concatenate X_train and y_train
    yy = y_train.reshape(1,-1)
    Xy_train = np.concatenate((X_train,yy.T), axis=1)
    return X_train, X_test, y_train, y_test, Xy_train

# locate the best matching prototype 
def get_best_matching_prototype (x,prototype_set):
    dis_list = list()
    for xyp in prototype_set:
        dis = np.linalg.norm(x-xyp[:-1]) 
        dis_list.append((xyp,dis))
    dis_list.sort(key=lambda tup: tup[1])
    return dis_list[0][0]

# # create a random prototype vector 
# def create_random_prototype(data_train):
#     n_records = len(data_train)
#     n_features = len(data_train[0])
#     xy_prototype = [data_train[np.random.randint(n_records)][i] for i in range(n_features)]
#     return xy_prototype

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
    for epoch in range(n_epochs):
        lrate = lrate_init * (1.0 - epoch/float(n_epochs))
        sum_err = 0.0
        for xy_train in data_train:
            bmu = get_best_matching_prototype(xy_train[:-1], prototype_set) # bmu is a view of the prototype_set
            err = xy_train[:-1] - bmu[:-1]
            sum_err += np.linalg.norm(err)**2 
            if bmu[-1] == xy_train[-1]: 
                bmu[:-1] += lrate*err 
            else:
                bmu[:-1] -= lrate*err 
        # print('>epoch=%d, lrate=%.3f, err=%.3f'%(epoch,lrate,sum_err))
    return prototype_set
    
# main(): 
test_sz = 75
lrate_init = 0.5
n_epochs = 30
# n_prototypes = 3 
n_run = 100
print('LVQ classifier, Iris dataset')
print('Train size:', len(iris_y)-test_sz, ', test size:', test_sz)
n_prototypes_vec = np.size(np.unique(y_test))*np.arange(1,10)
accuracy_vs_nprototype = []
stdaccuracy_vs_nprototypes = []
traintime_vs_nprototypes = []
testtime_vs_nprototypes = []
for n_prototypes in n_prototypes_vec:
    # refresh performance vectors
    accuracy_vec = np.zeros(n_run) 
    ttrain_vec = np.zeros(n_run)
    ttest_vec = np.zeros(n_run) 
    print('Number of prototypes = ',n_prototypes)
    for i_run in range(n_run):
        # split data set and test set 
        (X_train, X_test, y_train, y_test, Xy_train) = split_data(test_sz)
        
        # lvq training 
        ts_train = time()
        Xy_prototype = np.array(train_prototypes(Xy_train, n_prototypes, lrate_init, n_epochs))
        ttrain_vec[i_run] = time()-ts_train

        # predict outputs 
        predict_list = list()
        ts_test = time()
        for x_test in X_test:
            # 1NN 
            predict_list.append(get_best_matching_prototype(x_test,Xy_prototype)[-1])
        ttest_vec[i_run] = time()-ts_test
        y_predict = np.array(predict_list)

        # evaluate accuracy 
        e = 0
        for ii in range(len(y_predict)): 
            if y_predict[ii] != y_test[ii]:
                e += 1
        accuracy_vec[i_run] = 100*(1-e/len(y_test))
        #print('irun =', i_run, ', accuracy =', np.round(accuracy_vec[i_run],2))

    # print('accuracy:',np.round(accuracy_vec,2))
    # print('train time:',np.round(ttrain_vec,5))
    # print('test time:',np.round(ttest_vec,5))
    accuracy_avg = np.average(accuracy_vec)
    rtime_avg = np.average(ttest_vec)
    accuracy_var = np.var(accuracy_vec)
    traintime_avg = np.average(ttrain_vec)
    testtime_avg = np.average(ttest_vec)
    print('average accuracy: %.2f%%'%(np.average(accuracy_vec)))
    print('Accuracy variance:  %.2f %%' % accuracy_var)
    print('average train time: %.5fs'%(traintime_avg))
    print('average test time: %.5fs'%(testtime_avg))
    accuracy_vs_nprototype += [accuracy_avg]
    traintime_vs_nprototypes += [traintime_avg]
    testtime_vs_nprototypes += [testtime_avg]
    stdaccuracy_vs_nprototypes += [np.sqrt(accuracy_var)]

    # plot accurracy vs irun 
    # error figure
    plt.figure(figsize=(7, 6))
    plt.plot(accuracy_vec)
    plt.xlabel('Index of run')
    plt.ylabel('Accuracy (%)')
    plt.title('LVQ classifier, Iris dataset')
    plt.grid()
    plt.show()

    # confution matrix 
    cm = confusion_matrix(y_test,y_predict) 
    cm_df = pd.DataFrame(cm,
                            index=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'],
                            columns=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True)
    plt.title("LVQ Classifier, Iris dataset \nAvg. Accuracy: {:.2f}%, Avg. test time: {:.3f} (s)".format(accuracy_avg, rtime_avg))
    # plt.title('1NN Classifier, Iris dataset \nAccuracy: %.2f%, Running time: %.2f (s)'%(accuracy_avg, rtime_avg))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# plot accuracy vs no. of prototypes 
plt.figure(figsize=(7, 6))
plt.plot(n_prototypes_vec/np.size(np.unique(y_test)), accuracy_vs_nprototype)
plt.xlabel('Number of prototypes for each class')
plt.ylabel('Average accuracy (%)')
plt.title('LVQ classifier, Iris dataset')
plt.grid()
plt.show()

#plot standard deviation of accuracy vs. no. of prototypes
plt.figure(figsize=(7, 6))
plt.plot(n_prototypes_vec/np.size(np.unique(y_test)), stdaccuracy_vs_nprototypes)
plt.xlabel('Number of prototypes for each class')
plt.ylabel('Standard deviation of accuracy (%)')
plt.title('LVQ classifier, Iris dataset')
plt.grid()
plt.show()

# plot training time vs no. of prototypes 
plt.figure(figsize=(7, 6))
plt.plot(n_prototypes_vec/np.size(np.unique(y_test)), traintime_vs_nprototypes)
plt.xlabel('Number of prototypes for each class')
plt.ylabel('Training time (s)')
plt.title('LVQ classifier, Iris dataset')
plt.grid()
plt.show()

# plot test time vs no. of prototypes 
plt.figure(figsize=(7, 6))
plt.plot(n_prototypes_vec/np.size(np.unique(y_test)), testtime_vs_nprototypes)
plt.xlabel('Number of prototypes for each class')
plt.ylabel('Test time (s)')
plt.title('LVQ classifier, Iris dataset')
plt.grid()
plt.show()