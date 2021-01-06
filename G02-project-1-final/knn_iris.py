# m5232108 Hoang Tuan Linh

from __future__ import print_function
import numpy as np 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# np.random.seed(7)
print('1NN classifier, Iris dataset')
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target 
print('Labels:', np.unique(iris_y))

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

# main():
n_run = 100
test_sz = 75
accuracy_vec = list()
rtime_vec = list()
print('Train size:', len(iris_y)-test_sz, ', test size:', test_sz)
print('Number of runs:',n_run)
for i in range(n_run):
    # split training and testing sets 
    # X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=test_sz)
    (X_train, X_test, y_train, y_test, Xy_train) = split_data(test_sz)

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

    # evaluate accuracy and running time 
    accuracy_vec.append(100*accuracy_score(y_test, y_pred))
    

# print('Accuracy:',np.round(accuracy_vec,2))
accuracy_avg = np.average(accuracy_vec)
print('Average accuracy:  %.2f %%' % accuracy_avg)
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
plt.grid()
plt.show()

# confution matrix 
cm = confusion_matrix(y_test,y_pred) 
cm_df = pd.DataFrame(cm,
                        index=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'],
                        columns=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'])

plt.figure(figsize=(7, 6))
sns.heatmap(cm_df, annot=True)
plt.title("1NN Classifier, Iris dataset \nAvg. Accuracy: {:.2f}%, Avg. running time: {:.3f} (s)".format(accuracy_avg, rtime_avg))
# plt.title('1NN Classifier, Iris dataset \nAccuracy: %.2f%, Running time: %.2f (s)'%(accuracy_avg, rtime_avg))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()