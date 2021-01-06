import numpy as np
import _pickle as cPickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import losses
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
labels = [ 'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck']
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(6,6), title="Confusion matrix"):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    # cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_test,y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%d/%d\n%.1f%%' % (c, s, p)
            # elif c == 0:
            #     annot[i, j] = ''
            else:
                annot[i, j] = '%d\n%.1f%%' % (c,p)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True labels'
    cm.columns.name = 'Predicted labels'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig("")
    plt.title(title)
    plt.show()

def get_cifar10():
    def unpickle(file):
        fo = open(file, 'rb')
        dict = cPickle.load(fo, encoding='latin1')
        fo.close()
        return dict


    Batch_path = './dataset/CIFAR10/cifar-10-batches-py/'

    cifar10_batches = []
    cifar10_batches.append(unpickle(Batch_path+'data_batch_1'))
    cifar10_batches.append(unpickle(Batch_path+'data_batch_2'))
    cifar10_batches.append(unpickle(Batch_path+'data_batch_3'))
    cifar10_batches.append(unpickle(Batch_path+'data_batch_4'))
    cifar10_batches.append(unpickle(Batch_path+'data_batch_5'))

    cifar10_test_batch = unpickle(Batch_path+'test_batch')
    cifar10_test_data = cifar10_test_batch['data']
    cifar10_test_label = np.asarray(cifar10_test_batch['labels'])

    # print cifar10_batches[0]['data'].shape

    # stack the array all together
    cifar10_data = cifar10_batches[0]['data']
    cifar10_labels = cifar10_batches[0]['labels']

    for batch_idx in range(1,5):
        cifar10_data = np.vstack((cifar10_data,cifar10_batches[batch_idx]['data']))
        cifar10_labels += cifar10_batches[batch_idx]['labels']

    cifar10_labels = np.asarray(cifar10_labels)
    return cifar10_data, cifar10_labels, cifar10_test_data, cifar10_test_label

x_train, y_train, x_test, y_test = get_cifar10()
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=200, n_jobs=10)
#rf_model = BaggingClassifier(n_estimators=150, max_samples=0.1, max_features = 0.7)
#rf_model = AdaBoostClassifier(n_estimators= 150, learning_rate= 0.5)
test_time = time.time()
rf_model.fit(x_train,y_train)
end_time = time.time()
prediction = rf_model.predict(x_test)
cm = confusion_matrix(y_test, prediction)
cm_analysis(y_test, prediction, labels, ymap=None, figsize=(7,6), title="Ensemble: Random forest - CIFAR10 dataset")
acc_rf = rf_model.score(x_test, y_test)
f1_score = f1_score(y_test, prediction, zero_division=1, average='macro')
precision_score = precision_score(y_test, prediction, average='macro')
recall_score = recall_score(y_test, prediction, average='macro')
print(test_time - start_time)
print(end_time - test_time)
print(cm)
print(recall_score)
print(' Accuracy Score: ',acc_rf)
print(f1_score)
print(precision_score)
